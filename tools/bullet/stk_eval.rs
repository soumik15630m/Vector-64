// Gold oracle: load a stk_train checkpoint and print bullet's real raw graph
// output per FEN (eval_raw_output[0] = main + psq_stm - psq_ntm). Compared in
// Python against STKNet(transferred): STKNet_cp should equal bullet_raw * 400.
use bullet_lib::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::optimiser::AdamW,
    trainer::save::SavedFormat,
    value::ValueTrainerBuilder,
};
use bulletformat::ChessBoard;

const KING_BUCKETS: usize = 32;
const KINDS: usize = 11;
const SQUARES: usize = 64;
const NUM: usize = KING_BUCKETS * KINDS * SQUARES;
const DEAD: usize = NUM;

#[derive(Clone, Copy, Debug, Default)]
pub struct StkHalfKa;
#[inline]
fn flip_file(sq: usize) -> usize {
    sq ^ 7
}
#[inline]
fn flip_rank(sq: usize) -> usize {
    sq ^ 56
}
#[inline]
fn king_orient(ksq: usize) -> (usize, bool) {
    let mirror = (ksq & 7) >= 4;
    let tk = if mirror { flip_file(ksq) } else { ksq };
    ((tk >> 3) * 4 + (tk & 7), mirror)
}
impl SparseInputType for StkHalfKa {
    type RequiredDataType = ChessBoard;
    fn num_inputs(&self) -> usize {
        NUM + 1
    }
    fn max_active(&self) -> usize {
        32
    }
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let (mut sk, mut nk) = (0usize, 0usize);
        for (piece, square) in (*pos).into_iter() {
            if piece & 7 == 5 {
                if piece & 8 == 0 {
                    sk = square as usize;
                } else {
                    nk = square as usize;
                }
            }
        }
        let (sb, sm) = king_orient(sk);
        let (nb, nm) = king_orient(flip_rank(nk));
        for (piece, square) in (*pos).into_iter() {
            let c = usize::from(piece & 8 > 0);
            let pt = usize::from(piece & 7);
            let sq = usize::from(square);
            let ts = if sm { flip_file(sq) } else { sq };
            let stm = if c == 0 {
                if pt == 5 { DEAD } else { (sb * KINDS + pt) * SQUARES + ts }
            } else {
                (sb * KINDS + pt + 5) * SQUARES + ts
            };
            let sqn = flip_rank(sq);
            let tn = if nm { flip_file(sqn) } else { sqn };
            let ntm = if c == 1 {
                if pt == 5 { DEAD } else { (nb * KINDS + pt) * SQUARES + tn }
            } else {
                (nb * KINDS + pt + 5) * SQUARES + tn
            };
            f(stm, ntm);
        }
    }
    fn shorthand(&self) -> String {
        format!("stkhalfka-{NUM}")
    }
    fn description(&self) -> String {
        "STK-HalfKA".to_string()
    }
}

#[derive(Clone, Copy, Default)]
pub struct StkBucket;
impl OutputBuckets<ChessBoard> for StkBucket {
    const BUCKETS: usize = 8;
    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (((pos.occ().count_ones() as usize - 1) / 4).min(7)) as u8
    }
}

fn main() {
    const HL: usize = 1024;
    const L1: usize = 16;
    const L2: usize = 32;
    const NB: usize = 8;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(StkHalfKa)
        .output_buckets(StkBucket)
        .save_format(&[SavedFormat::id("l0w")])
        .loss_fn(|o, t| o.sigmoid().squared_error(t))
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", NUM + 1, HL);
            let l1 = builder.new_affine("l1", HL, NB * L1);
            let l2 = builder.new_affine("l2", L1, NB * L2);
            let l3 = builder.new_affine("l3", L2, NB);
            let psqt = builder.new_affine("psqt", NUM + 1, NB);
            let ft = |input, s, e| l0.slice(s, e).forward(input).crelu();
            let stm_h = ft(stm, 0, HL / 2) * ft(stm, HL / 2, HL);
            let ntm_h = ft(ntm, 0, HL / 2) * ft(ntm, HL / 2, HL);
            let hidden = stm_h.concat(ntm_h);
            let a1 = l1.forward(hidden).select(buckets).crelu();
            let a2 = l2.forward(a1).select(buckets).crelu();
            let main = l3.forward(a2).select(buckets);
            let psq_stm = psqt.forward(stm).select(buckets);
            let psq_ntm = psqt.forward(ntm).select(buckets);
            main + psq_stm - psq_ntm
        });

    trainer.load_from_checkpoint("checkpoints/stk-3");

    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 6 6",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    for (i, fen) in fens.iter().enumerate() {
        let raw = trainer.eval_raw_output(fen);
        println!("EVAL{i} {} {}", raw[0], fen);
    }
}

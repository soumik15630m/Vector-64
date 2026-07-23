// Feature-parity harness for the custom STK-HalfKA input type.
// Prints, per FEN, the sorted active feature indices for the STM and NTM
// perspectives (own king dropped) so they can be diffed against
// tools/nnue/halfka_features.py. This is the phase-1 gate for the bullet port.
use bulletformat::ChessBoard;
use bullet_lib::game::inputs::SparseInputType;

const KING_BUCKETS: usize = 32;
const KINDS: usize = 11; // own {P,N,B,R,Q}=0..4, enemy {P,N,B,R,Q,K}=5..10
const SQUARES: usize = 64;
const NUM: usize = KING_BUCKETS * KINDS * SQUARES; // 22528
const DEAD: usize = NUM; // throwaway slot for the own king

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

// bucket + mirror flag for a king already oriented to its perspective's bottom
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
        // Kings straight from the piece list (color 0 = mover/stm, 1 = opp/ntm).
        let (mut stm_king, mut ntm_king) = (0usize, 0usize);
        for (piece, square) in (*pos).into_iter() {
            if piece & 7 == 5 {
                if piece & 8 == 0 {
                    stm_king = square as usize;
                } else {
                    ntm_king = square as usize;
                }
            }
        }
        // STM sees the board as-is (mover at bottom); NTM rank-flips it.
        let (stm_bucket, stm_mirror) = king_orient(stm_king);
        let (ntm_bucket, ntm_mirror) = king_orient(flip_rank(ntm_king));

        for (piece, square) in (*pos).into_iter() {
            let c = usize::from(piece & 8 > 0); // 0 = mover, 1 = opp
            let pt = usize::from(piece & 7); // 0=P..5=K
            let sq = usize::from(square);

            let ts = if stm_mirror { flip_file(sq) } else { sq };
            let stm = if c == 0 {
                if pt == 5 { DEAD } else { (stm_bucket * KINDS + pt) * SQUARES + ts }
            } else {
                (stm_bucket * KINDS + pt + 5) * SQUARES + ts
            };

            let sqn = flip_rank(sq);
            let tn = if ntm_mirror { flip_file(sqn) } else { sqn };
            let ntm = if c == 1 {
                if pt == 5 { DEAD } else { (ntm_bucket * KINDS + pt) * SQUARES + tn }
            } else {
                (ntm_bucket * KINDS + pt + 5) * SQUARES + tn
            };

            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        format!("stkhalfka-{NUM}")
    }
    fn description(&self) -> String {
        "STK-HalfKA 32kb x 11 kinds x 64, own king dropped".to_string()
    }
}

fn dump(tag: &str, fen: &str) {
    let board: ChessBoard = format!("{fen} | 0 | 0.5").parse().unwrap();
    let (mut stm, mut ntm) = (Vec::new(), Vec::new());
    StkHalfKa.map_features(&board, |s, n| {
        if s != DEAD {
            stm.push(s);
        }
        if n != DEAD {
            ntm.push(n);
        }
    });
    stm.sort_unstable();
    ntm.sort_unstable();
    let j = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    println!("{tag} STM {}", j(&stm));
    println!("{tag} NTM {}", j(&ntm));
}

fn main() {
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 6",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 6 6",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    for (i, fen) in fens.iter().enumerate() {
        dump(&format!("FEN{i}"), fen);
    }
}

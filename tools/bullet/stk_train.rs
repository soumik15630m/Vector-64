// STK-HalfKA trainer in bullet. Mirrors tools/nnue/make_net.py's STKNet:
//   FT 22528(+dead)->1024 per persp, CReLU, pairwise-mul (512+512),
//   per-bucket L1 1024->16 (CReLU), L2 16->32 (CReLU), out 32->1,
//   plus PSQT (feature->8, psq_stm - psq_ntm). 8 output buckets = (pieces-1)/4.
// Saves raw f32 weights for transfer into STKNet (bit-exact export reused there).
use bullet_lib::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use bulletformat::ChessBoard;

// ------------------- custom input type (verified in stk_halfka.rs) -------------------
const KING_BUCKETS: usize = 32;
const KINDS: usize = 11;
const SQUARES: usize = 64;
const NUM: usize = KING_BUCKETS * KINDS * SQUARES; // 22528
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
        "STK-HalfKA 32kb x 11 kinds x 64, own king dropped".to_string()
    }
}

// ------------------- custom output buckets: (pieces - 1) / 4, clamped [0,7] -------------------
#[derive(Clone, Copy, Default)]
pub struct StkBucket;
impl OutputBuckets<ChessBoard> for StkBucket {
    const BUCKETS: usize = 8;
    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (((pos.occ().count_ones() as usize - 1) / 4).min(7)) as u8
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = args.get(1).cloned().unwrap_or_else(|| "data/stk.bin".to_string());
    let sbs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let bps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6104);

    const HL: usize = 1024;
    const L1: usize = 16;
    const L2: usize = 32;
    const NB: usize = 8;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(StkHalfKa)
        .output_buckets(StkBucket)
        // raw f32 weights -> read by the STKNet transfer script
        .save_format(&[
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
            SavedFormat::id("psqtw"),
            SavedFormat::id("psqtb"),
        ])
        .loss_fn(|o, t| o.sigmoid().squared_error(t))
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", NUM + 1, HL);
            let l1 = builder.new_affine("l1", HL, NB * L1);
            let l2 = builder.new_affine("l2", L1, NB * L2);
            let l3 = builder.new_affine("l3", L2, NB);
            let psqt = builder.new_affine("psqt", NUM + 1, NB);

            // FT -> CReLU -> pairwise-mul within each perspective (512*512 -> 512)
            let ft = |input, s, e| l0.slice(s, e).forward(input).crelu();
            let stm_h = ft(stm, 0, HL / 2) * ft(stm, HL / 2, HL);
            let ntm_h = ft(ntm, 0, HL / 2) * ft(ntm, HL / 2, HL);
            let hidden = stm_h.concat(ntm_h); // 1024

            let a1 = l1.forward(hidden).select(buckets).crelu();
            let a2 = l2.forward(a1).select(buckets).crelu();
            let main = l3.forward(a2).select(buckets);

            let psq_stm = psqt.forward(stm).select(buckets);
            let psq_ntm = psqt.forward(ntm).select(buckets);
            main + psq_stm - psq_ntm
        });

    let schedule = TrainingSchedule {
        net_id: "stk".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: bps,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        lr_scheduler: lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3f32.powi(5), final_superbatch: sbs },
        save_rate: 5,
    };

    let settings =
        LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };
    let dataloader = DirectSequentialDataLoader::new(&[&data]);
    trainer.run(&schedule, &settings, &dataloader);
}

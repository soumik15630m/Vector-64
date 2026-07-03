#include "search.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>

namespace Search {
    namespace {

        // Static exchange evaluation (swap algorithm). Returns the expected
        // material gain of the capture from the mover's point of view.
        int see(const Core::Position& pos, Core::Move m) {
            const Core::Square to = m.to_sq();
            const Core::Square from = m.from_sq();

            Core::Bitboard occ = pos.occupancy();
            Core::Color stm = pos.side_to_move();

            int gain[40];
            int d = 0;

            const Core::PieceType victim = m.is_en_passant() ? Core::PAWN : pos.piece_on(to);
            gain[0] = Core::PieceValue[victim];

            Core::PieceType onTarget = pos.piece_on(from);
            occ ^= Core::square_bb(from);
            if (m.is_en_passant()) {
                occ ^= Core::square_bb(Core::make_square(
                    static_cast<Core::GenFile>(Core::file_of(to)),
                    static_cast<Core::GenRank>(Core::rank_of(from))));
            }

            stm = ~stm;
            Core::Bitboard attackers = pos.attackers_to(to, occ) & occ;

            // Diagonal/orthogonal slider sets (both colors). After removing an
            // attacker we only re-scan the ray it could have blocked, instead
            // of rebuilding the whole attacker set — same values, far fewer
            // lookups. Node counts confirm behavioural equivalence.
            const Core::Bitboard bishopsQueens = pos.pieces(Core::BISHOP) | pos.pieces(Core::QUEEN);
            const Core::Bitboard rooksQueens = pos.pieces(Core::ROOK) | pos.pieces(Core::QUEEN);

            while (attackers & pos.pieces(stm)) {
                const Core::Bitboard stmAttackers = attackers & pos.pieces(stm);

                // Least valuable attacker recaptures first.
                Core::PieceType pt = Core::PAWN;
                Core::Bitboard sub = 0;
                for (int t = Core::PAWN; t <= Core::KING; ++t) {
                    sub = stmAttackers & pos.pieces(static_cast<Core::PieceType>(t));
                    if (sub) {
                        pt = static_cast<Core::PieceType>(t);
                        break;
                    }
                }

                // A king cannot recapture a defended piece.
                if (pt == Core::KING && (attackers & pos.pieces(~stm) & ~stmAttackers)) break;

                d++;
                gain[d] = Core::PieceValue[onTarget] - gain[d - 1];
                onTarget = pt;
                occ ^= Core::square_bb(Core::lsb(sub));

                // Reveal any slider that was hiding behind the piece we removed.
                if (pt == Core::PAWN || pt == Core::BISHOP || pt == Core::QUEEN) {
                    attackers |= Core::Attacks::bishop_attacks(to, occ) & bishopsQueens;
                }
                if (pt == Core::ROOK || pt == Core::QUEEN) {
                    attackers |= Core::Attacks::rook_attacks(to, occ) & rooksQueens;
                }
                attackers &= occ;

                stm = ~stm;
                if (d >= 38) break;
            }

            while (d > 0) {
                gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
                --d;
            }
            return gain[0];
        }

        int piece_order(Core::PieceType pt) {
            return static_cast<int>(pt);  // NO_PIECE_TYPE=0, PAWN..KING ascending
        }

        // Stable lazy selection: bring the highest-scored remaining move to
        // `idx` (preserving generation order among equal scores) and return
        // it. At cut-nodes only a couple of picks are ever paid for.
        Core::Move stable_pick(Core::MoveList& moves, int* scores, int idx) {
            int best = idx;
            for (int i = idx + 1; i < moves.size(); ++i) {
                if (scores[i] > scores[best]) best = i;
            }
            const Core::Move m = moves.moves[best];
            const int s = scores[best];
            for (int i = best; i > idx; --i) {
                moves.moves[i] = moves.moves[i - 1];
                scores[i] = scores[i - 1];
            }
            moves.moves[idx] = m;
            scores[idx] = s;
            return m;
        }

        // Staged move emission for the main search: TT move with no
        // generation at all, then captures/promotions by MVV-LVA, killers,
        // then quiets by history. In check it falls back to scored legal
        // evasions. Emits pseudo-legal moves unless emits_legal() is true.
        class StagedMoves {
        public:
            StagedMoves(Core::Position& pos, const Core::NodeLegality& nl, Core::Move ttMove,
                        const MoveOrdering& ordering, int ply)
                : pos_(pos), nl_(nl), ttMove_(ttMove), ordering_(ordering), ply_(ply) {
                stage_ = nl.in_check() ? EVASION_GEN : TT_MOVE;
            }

            bool emits_legal() const { return nl_.in_check(); }

            Core::Move next() {
                switch (stage_) {
                case TT_MOVE:
                    stage_ = CAP_GEN;
                    if (ttMove_.is_ok() && Core::is_pseudo_legal(pos_, ttMove_)) return ttMove_;
                    [[fallthrough]];
                case CAP_GEN:
                    Core::generate_pseudo_captures(pos_, list_);
                    for (int i = 0; i < list_.size(); ++i) {
                        const Core::Move m = list_[i];
                        const Core::PieceType victim =
                            m.is_en_passant() ? Core::PAWN : pos_.piece_on(m.to_sq());
                        int s = 16 * piece_order(victim) - piece_order(pos_.piece_on(m.from_sq()));
                        if (m.is_promotion()) s += 16 * piece_order(m.promotion_type());
                        scores_[i] = s;
                    }
                    idx_ = 0;
                    stage_ = CAPS;
                    [[fallthrough]];
                case CAPS:
                    while (idx_ < list_.size()) {
                        const Core::Move m = stable_pick(list_, scores_, idx_++);
                        if (m != ttMove_) return m;
                    }
                    stage_ = KILLER_1;
                    [[fallthrough]];
                case KILLER_1: {
                    stage_ = KILLER_2;
                    const Core::Move k = ordering_.killer(ply_, 0);
                    if (k.is_ok() && k != ttMove_ && Core::is_pseudo_legal(pos_, k)) {
                        killer1_ = k;
                        return k;
                    }
                }
                    [[fallthrough]];
                case KILLER_2: {
                    stage_ = QUIET_GEN;
                    const Core::Move k = ordering_.killer(ply_, 1);
                    if (k.is_ok() && k != ttMove_ && k != killer1_ && Core::is_pseudo_legal(pos_, k)) {
                        killer2_ = k;
                        return k;
                    }
                }
                    [[fallthrough]];
                case QUIET_GEN:
                    Core::generate_pseudo_quiets(pos_, list_);
                    for (int i = 0; i < list_.size(); ++i) {
                        scores_[i] = ordering_.history_score(pos_.side_to_move(), list_[i]);
                    }
                    idx_ = 0;
                    stage_ = QUIETS;
                    [[fallthrough]];
                case QUIETS:
                    while (idx_ < list_.size()) {
                        const Core::Move m = stable_pick(list_, scores_, idx_++);
                        if (m != ttMove_ && m != killer1_ && m != killer2_) return m;
                    }
                    stage_ = DONE;
                    return Core::Move::none();
                case EVASION_GEN:
                    Core::generate_legal_moves(pos_, list_);
                    for (int i = 0; i < list_.size(); ++i) {
                        scores_[i] = ordering_.score_move(pos_, list_[i], ttMove_, ply_);
                    }
                    idx_ = 0;
                    stage_ = EVASIONS;
                    [[fallthrough]];
                case EVASIONS:
                    if (idx_ < list_.size()) return stable_pick(list_, scores_, idx_++);
                    stage_ = DONE;
                    return Core::Move::none();
                default:
                    return Core::Move::none();
                }
            }

        private:
            enum Stage { TT_MOVE, CAP_GEN, CAPS, KILLER_1, KILLER_2, QUIET_GEN, QUIETS, EVASION_GEN, EVASIONS, DONE };

            Core::Position& pos_;
            const Core::NodeLegality& nl_;
            Core::Move ttMove_;
            const MoveOrdering& ordering_;
            int ply_;
            Stage stage_;
            Core::Move killer1_ = Core::Move::none();
            Core::Move killer2_ = Core::Move::none();
            Core::MoveList list_;
            int scores_[Core::MoveList::MAX_MOVES];
            int idx_ = 0;
        };
    }

    EngineSearch::EngineSearch(size_t hashMb)
        : hashMb_(hashMb),
          ownTt_(hashMb),
          tt_(&ownTt_),
          eval_(&evaluator_) {
    }

    EngineSearch::EngineSearch(TranspositionTable* sharedTt, const Evaluator* sharedEval)
        : hashMb_(1),
          ownTt_(1),
          tt_(sharedTt),
          eval_(sharedEval) {
    }

    void EngineSearch::set_hash_mb(size_t hashMb) {
        hashMb_ = std::max<size_t>(1, hashMb);
        tt_->resize_mb(hashMb_);
    }

    void EngineSearch::set_threads(int threads) {
        threadCount_ = std::clamp(threads, 1, 64);
    }

    void EngineSearch::clear() {
        tt_->clear();
        ordering_.clear();
    }

    bool EngineSearch::load_nnue(const std::string& path) {
        return evaluator_.load_nnue(path);
    }

    int EngineSearch::evaluate(const Core::Position& pos) {
        return eval_->evaluate(pos);
    }

    bool EngineSearch::check_stop(const Limits& limits, const Callbacks& callbacks) {
        if (stopped_) return true;
        if ((callbacks.shouldStop && callbacks.shouldStop()) ||
            (limits.maxNodes > 0 && nodes_ >= limits.maxNodes) ||
            (limits.hasDeadline && Clock::now() >= limits.deadline)) {
            stopped_ = true;
        }
        return stopped_;
    }

    void EngineSearch::update_pv(int ply, Core::Move move) {
        pvTable_[ply][0] = move;
        const int childLen = pvLen_[ply + 1];
        for (int i = 0; i < childLen; ++i) pvTable_[ply][i + 1] = pvTable_[ply + 1][i];
        pvLen_[ply] = childLen + 1;
    }

    // Continues searching captures at depth <= 0 to prevent the "horizon effect"
    // (e.g., stopping search while a queen is en prise).
    HOT_FN int EngineSearch::quiescence(
        Core::Position& pos,
        int alpha,
        int beta,
        int ply,
        const Limits& limits,
        const Callbacks& callbacks
    ) {
        pvLen_[ply] = 0;

        nodes_++;
        if ((nodes_ & 2047) == 0 && check_stop(limits, callbacks)) return 0;
        if (stopped_) return 0;

        if (ply > seldepth_) seldepth_ = ply;
        if (ply >= MAX_PLY) return eval_->evaluate(pos);

        const uint64_t key = pos.hash();
        const TTEntry* tte = tt_->probe(key);
        Core::Move ttMove = Core::Move::none();

        qProbes_++;
        if (tte != nullptr) {
            qHits_++;
            ttMove = tte->move();
            const int ttScore = TranspositionTable::scoreFromTT(tte->score, ply);

            if (tte->bound() == BOUND_EXACT)                     return ttScore;
            if (tte->bound() == BOUND_LOWER && ttScore >= beta)  return ttScore;
            if (tte->bound() == BOUND_UPPER && ttScore <= alpha) return ttScore;
        }

        const Core::NodeLegality nodeLegal = Core::make_node_legality(pos);
        const bool inCheck = nodeLegal.in_check();
        const int originalAlpha = alpha;

        int standPat = -INF_SCORE;
        if (!inCheck) {
            standPat = (tte != nullptr && tte->eval != TT_EVAL_NONE)
                ? tte->eval
                : eval_->evaluate(pos);

            if (standPat >= beta) {
                tt_->store(key, standPat, Core::Move::none(), 0, BOUND_LOWER, ply, standPat);
                return standPat;
            }
            if (standPat > alpha) alpha = standPat;

            // Delta Pruning
            const int DELTA = 975;
            if (standPat + DELTA < alpha) {
                tt_->store(key, alpha, Core::Move::none(), 0, BOUND_UPPER, ply, standPat);
                return alpha;
            }
        }
        const int qEval = inCheck ? TT_EVAL_NONE : standPat;

        Core::MoveList moves;
        bool movesAreLegal;
        if (inCheck) {
            // Evasions: search every legal reply, no stand pat while in check.
            Core::generate_legal_moves(pos, moves);
            if (moves.size() == 0) return -MATE_SCORE + ply;
            movesAreLegal = true;
        } else {
            Core::generate_pseudo_captures(pos, moves);
            movesAreLegal = false;
        }

        int scores[Core::MoveList::MAX_MOVES];
        for (int i = 0; i < moves.size(); ++i) {
            scores[i] = ordering_.score_move(pos, moves[i], ttMove, ply);
        }

        Core::Move bestMove = Core::Move::none();
        int bestScore = standPat;

        for (int i = 0; i < moves.size(); ++i) {
            const Core::Move move = stable_pick(moves, scores, i);
            if (!movesAreLegal && !Core::is_legal(nodeLegal, move)) continue;

            // Skip captures that lose material outright. When the victim is
            // worth at least the attacker, SEE >= 0 is guaranteed — skip the
            // swap loop, the pruning decision is identical.
            if (!inCheck && !move.is_promotion()) {
                const Core::PieceType victim =
                    move.is_en_passant() ? Core::PAWN : pos.piece_on(move.to_sq());
                const Core::PieceType attacker = pos.piece_on(move.from_sq());
                if (Core::PieceValue[victim] < Core::PieceValue[attacker] &&
                    see(pos, move) < 0) {
                    continue;
                }
            }

            Core::UndoInfo undo{};
            pos.make_move(move, undo);
            tt_->prefetch(pos.hash());
            const int score = -quiescence(pos, -beta, -alpha, ply + 1, limits, callbacks);
            pos.unmake_move(move, undo);

            if (stopped_) return 0;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
            if (score >= beta) {
                tt_->store(key, score, move, 0, BOUND_LOWER, ply, qEval);
                return score;
            }
            if (score > alpha) {
                alpha = score;
                update_pv(ply, move);
            }
        }

        const TTBound bound = (bestScore <= originalAlpha) ? BOUND_UPPER : BOUND_EXACT;
        tt_->store(key, bestScore, bestMove, 0, bound, ply, qEval);

        return bestScore;
    }

    HOT_FN int EngineSearch::negamax(
        Core::Position& pos,
        int depth,
        int alpha,
        int beta,
        int ply,
        const Limits& limits,
        const Callbacks& callbacks
    ) {
        if (depth <= 0) return quiescence(pos, alpha, beta, ply, limits, callbacks);

        pvLen_[ply] = 0;

        nodes_++;
        if ((nodes_ & 2047) == 0 && check_stop(limits, callbacks)) return 0;
        if (stopped_) return 0;

        // Draws by repetition or the fifty-move rule.
        if (pos.is_repetition() || pos.halfmove_clock() >= 100) return 0;
        if (ply >= MAX_PLY) return eval_->evaluate(pos);

        const bool isPVNode = (beta - alpha) > 1;

        const uint64_t key = pos.hash();
        const TTEntry* tte = tt_->probe(key);
        Core::Move ttMove = Core::Move::none();

        nProbes_++;
        if (tte != nullptr) {
            nHits_++;
            ttMove = tte->move();

            if (!isPVNode && tte->depth >= depth) {
                const int ttScore = TranspositionTable::scoreFromTT(tte->score, ply);

                if (tte->bound() == BOUND_EXACT)                     return ttScore;
                if (tte->bound() == BOUND_LOWER && ttScore >= beta)  return ttScore;
                if (tte->bound() == BOUND_UPPER && ttScore <= alpha) return ttScore;
            }
        }

        const Core::NodeLegality nodeLegal = Core::make_node_legality(pos);
        const bool inCheck = nodeLegal.in_check();

        int staticEval;
        if (inCheck) {
            staticEval = -INF_SCORE;
        } else if (tte != nullptr && tte->eval != TT_EVAL_NONE) {
            staticEval = tte->eval;
        } else {
            staticEval = eval_->evaluate(pos);
        }

        // When the TT score bounds the true value from the right side, it is
        // a better estimate than the static eval for the pruning decisions.
        int evalForPruning = staticEval;
        if (!inCheck && tte != nullptr) {
            const int ttScore = TranspositionTable::scoreFromTT(tte->score, ply);
            if (!is_mate_score(ttScore) &&
                (tte->bound() == BOUND_EXACT ||
                 (tte->bound() == BOUND_LOWER && ttScore > evalForPruning) ||
                 (tte->bound() == BOUND_UPPER && ttScore < evalForPruning))) {
                evalForPruning = ttScore;
            }
        }

        // Reverse futility pruning: far enough above beta that a shallow
        // search is extremely unlikely to fall back under it.
        if (!isPVNode && !inCheck && depth <= 8 && beta < MATE_BOUND &&
            evalForPruning - 80 * depth >= beta) {
            return evalForPruning;
        }

        // Null Move Pruning: give the opponent a free move; if our position
        // still beats beta, trust the fail-high. Skipped without non-pawn
        // material where zugzwang would make it unsound.
        if (!isPVNode && !inCheck && depth >= 3 && evalForPruning >= beta &&
            pos.has_non_pawn_material(pos.side_to_move())) {
            const int R = 3 + depth / 6;

            Core::UndoInfo undo;
            pos.make_null_move(undo);
            const int nullScore = -negamax(pos, depth - 1 - R, -beta, -beta + 1, ply + 1, limits, callbacks);
            pos.unmake_null_move(undo);

            if (stopped_) return 0;
            if (nullScore >= beta) return is_mate_score(nullScore) ? beta : nullScore;
        }

        StagedMoves picker(pos, nodeLegal, ttMove, ordering_, ply);
        const bool pickerEmitsLegal = picker.emits_legal();

        const Core::Color us = pos.side_to_move();
        const int originalAlpha = alpha;
        const int extension = inCheck ? 1 : 0;

        Core::Move bestMove = Core::Move::none();
        int bestScore = -INF_SCORE;
        int movesSearched = 0;
        Core::Move quietsTried[64];
        int quietsCount = 0;

        while (true) {
            const Core::Move move = picker.next();
            if (!move.is_ok()) break;
            if (!pickerEmitsLegal && !Core::is_legal(nodeLegal, move)) continue;

            const bool isQuiet = !move.is_capture() && !move.is_promotion();
            if (movesSearched == 0) bestMove = move;

            Core::UndoInfo undo{};
            pos.make_move(move, undo);
            tt_->prefetch(pos.hash());

            const int newDepth = depth - 1 + extension;
            int score;
            if (movesSearched == 0) {
                score = -negamax(pos, newDepth, -beta, -alpha, ply + 1, limits, callbacks);
            } else {
                // Late Move Reductions for quiet moves ordered far down the list.
                int reduction = 0;
                if (depth >= 3 && movesSearched >= 3 && isQuiet && !inCheck) {
                    reduction = 1 + (movesSearched >= 6 ? 1 : 0) + (depth >= 8 ? 1 : 0);
                    reduction = std::min(reduction, newDepth - 1);
                }

                score = -negamax(pos, newDepth - reduction, -alpha - 1, -alpha, ply + 1, limits, callbacks);

                // Reduced search beat alpha: verify at full depth, still null window.
                if (score > alpha && reduction > 0) {
                    score = -negamax(pos, newDepth, -alpha - 1, -alpha, ply + 1, limits, callbacks);
                }
                // Full window re-search only opens up inside PV nodes.
                if (score > alpha && score < beta) {
                    score = -negamax(pos, newDepth, -beta, -alpha, ply + 1, limits, callbacks);
                }
            }

            pos.unmake_move(move, undo);

            if (stopped_) return 0;
            movesSearched++;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha) {
                alpha = score;
                if (isPVNode) update_pv(ply, move);
            }

            if (alpha >= beta) {
                if (isQuiet) {
                    ordering_.update_killers(ply, move);
                    ordering_.update_history(us, move, depth);
                    for (int q = 0; q < quietsCount; ++q) {
                        ordering_.update_history_malus(us, quietsTried[q], depth);
                    }
                }
                break;
            }

            if (isQuiet && quietsCount < 64) quietsTried[quietsCount++] = move;
        }

        if (movesSearched == 0) {
            return inCheck ? (-MATE_SCORE + ply) : 0;
        }

        if (!stopped_) {
            TTBound bound;
            if (bestScore <= originalAlpha) bound = BOUND_UPPER;
            else if (bestScore >= beta)     bound = BOUND_LOWER;
            else                            bound = BOUND_EXACT;

            tt_->store(key, bestScore, bestMove, depth, bound, ply,
                       inCheck ? TT_EVAL_NONE : staticEval);
        }

        return bestScore;
    }

    int EngineSearch::search_root(
        Core::Position& root,
        Core::MoveList& rootMoves,
        int depth,
        int alpha,
        int beta,
        Core::Move prevBest,
        const Limits& limits,
        const Callbacks& callbacks
    ) {
        pvLen_[0] = 0;

        const TTEntry* tte = tt_->probe(root.hash());
        const Core::Move ttMove = tte ? tte->move() : Core::Move::none();
        ordering_.sort_moves(root, rootMoves, prevBest.is_ok() ? prevBest : ttMove, 0);

        const bool inCheck = root.in_check();
        const int originalAlpha = alpha;
        const int extension = inCheck ? 1 : 0;

        Core::Move bestMove = rootMoves[0];
        int bestScore = -INF_SCORE;

        for (int i = 0; i < rootMoves.size(); ++i) {
            const Core::Move move = rootMoves[i];

            Core::UndoInfo undo{};
            root.make_move(move, undo);
            tt_->prefetch(root.hash());

            const int newDepth = depth - 1 + extension;
            int score;
            if (i == 0) {
                score = -negamax(root, newDepth, -beta, -alpha, 1, limits, callbacks);
            } else {
                score = -negamax(root, newDepth, -alpha - 1, -alpha, 1, limits, callbacks);
                if (score > alpha && score < beta) {
                    score = -negamax(root, newDepth, -beta, -alpha, 1, limits, callbacks);
                }
            }

            root.unmake_move(move, undo);

            if (stopped_) return 0;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha) {
                alpha = score;
                update_pv(0, move);
            }

            if (alpha >= beta) break;
        }

        if (!stopped_) {
            TTBound bound;
            if (bestScore <= originalAlpha) bound = BOUND_UPPER;
            else if (bestScore >= beta)     bound = BOUND_LOWER;
            else                            bound = BOUND_EXACT;

            tt_->store(root.hash(), bestScore, bestMove, depth, bound, 0, TT_EVAL_NONE);
        }

        return bestScore;
    }

    uint64_t EngineSearch::total_nodes() const {
        uint64_t total = nodes_;
        // Helper counters are read racily; this is a reporting statistic.
        for (const EngineSearch* h : helperViews_) total += h->nodes_;
        return total;
    }

    // Lazy SMP: helper threads run the same iterative deepening on their own
    // move ordering state while sharing the transposition table. Entries may
    // tear under concurrent writes; the 32-bit key check plus legal-move
    // matching keeps a torn entry from doing more than costing a few nodes.
    Result EngineSearch::search(Core::Position root, const Limits& limits, const Callbacks& callbacks) {
        tt_->new_search();
        helperViews_.clear();

        if (threadCount_ <= 1) {
            return search_internal(root, limits, callbacks);
        }

        std::atomic<bool> helpersStop{false};
        std::vector<std::unique_ptr<EngineSearch>> helpers;
        std::vector<std::thread> threads;
        helpers.reserve(threadCount_ - 1);

        for (int i = 1; i < threadCount_; ++i) {
            helpers.emplace_back(new EngineSearch(tt_, eval_));
            helperViews_.push_back(helpers.back().get());
        }

        Callbacks helperCallbacks;
        helperCallbacks.shouldStop = [&helpersStop]() {
            return helpersStop.load(std::memory_order_relaxed);
        };

        for (auto& helper : helpers) {
            // Copy the position here, on this thread: the master starts
            // mutating `root` via make/unmake as soon as the loop ends.
            threads.emplace_back([&limits, &helperCallbacks, h = helper.get(), pos = root]() mutable {
                h->search_internal(pos, limits, helperCallbacks);
            });
        }

        Result out = search_internal(root, limits, callbacks);

        helpersStop.store(true, std::memory_order_relaxed);
        for (std::thread& t : threads) t.join();

        out.nodes = total_nodes();
        helperViews_.clear();
        return out;
    }

    Result EngineSearch::search_internal(Core::Position& root, const Limits& limits, const Callbacks& callbacks) {
        Result out{};
        nodes_ = 0;
        seldepth_ = 0;
        stopped_ = false;
        started_ = Clock::now();

        ordering_.clear();
        ordering_.age_history();

        Core::MoveList rootMoves;
        Core::generate_legal_moves(root, rootMoves);

        if (limits.searchMoves.size() > 0) {
            Core::MoveList filtered;
            for (int i = 0; i < rootMoves.size(); ++i) {
                for (int k = 0; k < limits.searchMoves.size(); ++k) {
                    if (rootMoves[i] == limits.searchMoves[k]) {
                        filtered.push_back(rootMoves[i]);
                        break;
                    }
                }
            }
            if (filtered.size() > 0) rootMoves = filtered;
        }

        if (rootMoves.size() == 0) {
            out.bestMove = Core::Move::none();
            out.scoreCp = root.in_check() ? -MATE_SCORE : 0;
            out.nodes = nodes_;
            return out;
        }

        out.bestMove = rootMoves[0];
        int prevScore = 0;

        for (int depth = 1; depth <= limits.maxDepth; ++depth) {
            if (check_stop(limits, callbacks)) break;

            qProbes_ = 0;
            qHits_ = 0;
            nProbes_ = 0;
            nHits_ = 0;

            // Aspiration window around the previous iteration's score.
            int delta = 25;
            int alpha = -INF_SCORE;
            int beta = INF_SCORE;
            if (depth >= 4) {
                alpha = std::max(-INF_SCORE, prevScore - delta);
                beta = std::min(INF_SCORE, prevScore + delta);
            }

            int score;
            while (true) {
                score = search_root(root, rootMoves, depth, alpha, beta, out.bestMove, limits, callbacks);
                if (stopped_) break;

                if (score <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = std::max(-INF_SCORE, score - delta);
                } else if (score >= beta) {
                    beta = std::min(INF_SCORE, score + delta);
                } else {
                    break;
                }
                delta += delta / 2 + 5;
            }

            if (stopped_) break;  // discard the partial iteration

            if (pvLen_[0] > 0) out.bestMove = pvTable_[0][0];
            out.scoreCp = score;
            out.completedDepth = depth;
            out.nodes = nodes_;
            prevScore = score;

            if (callbacks.onInfo) {
                IterInfo info;
                info.depth = depth;
                info.seldepth = seldepth_;
                info.scoreCp = score;
                info.nodes = total_nodes();
                info.elapsedMs = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started_).count()
                );
                info.pv = pvTable_[0];
                info.pvLen = pvLen_[0];
                info.qsearchTtHitRate = qProbes_ > 0 ? (100.0 * qHits_ / qProbes_) : 0.0;
                info.negamaxTtHitRate = nProbes_ > 0 ? (100.0 * nHits_ / nProbes_) : 0.0;
                callbacks.onInfo(info);
            }
        }

        // If even depth 1 didn't finish (very tight deadline), fall back to
        // the first legal move with a static eval score.
        if (out.completedDepth == 0) {
            out.scoreCp = eval_->evaluate(root);
        }
        out.nodes = nodes_;

        return out;
    }
}

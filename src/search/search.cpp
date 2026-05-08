#include "search.h"

#include <algorithm>
#include <iostream>
#include <atomic>

namespace Search {
    namespace {
        uint64_t q_probes = 0;
        uint64_t q_hits = 0;
        uint64_t n_probes = 0;
        uint64_t n_hits = 0;
    }

    EngineSearch::EngineSearch(size_t hashMb)
        : hashMb_(hashMb),
          tt_(hashMb) {
    }

    void EngineSearch::set_hash_mb(size_t hashMb) {
        hashMb_ = std::max<size_t>(1, hashMb);
        tt_.resize_mb(hashMb_);
    }

    void EngineSearch::clear() {
        tt_.clear();
        ordering_.clear();
    }

    bool EngineSearch::load_nnue(const std::string& path) {
        return evaluator_.load_nnue(path);
    }

    bool EngineSearch::should_stop(const Limits& limits, const Callbacks& callbacks) const {
        if (callbacks.shouldStop && callbacks.shouldStop()) return true;
        if (limits.maxNodes > 0 && nodes_ >= limits.maxNodes) return true;
        if (limits.hasDeadline && Clock::now() >= limits.deadline) return true;
        return false;
    }

    // Continues searching captures at depth <= 0 to prevent the "horizon effect"
    // (e.g., stopping search while a queen is en prise).
    int EngineSearch::quiescence(
        Core::Position& pos,
        int alpha,
        int beta,
        int ply,
        const Limits& limits,
        const Callbacks& callbacks
    ) {
        if ((nodes_ & 2047) == 0 && should_stop(limits, callbacks)) return 0;
        nodes_++;

        uint64_t key = pos.hash();
        const TTEntry* tte = tt_.probe(key);
        Core::Move ttMove = Core::Move::none();

        q_probes++;
        if (tte != nullptr) {
            q_hits++;
        }

        if (tte != nullptr) {
            ttMove = tte->move;
            int ttScore = TranspositionTable::scoreFromTT(tte->score, ply);

            if (tte->depth >= 0) {
                if (tte->bound == BOUND_EXACT)                       return ttScore;
                if (tte->bound == BOUND_LOWER && ttScore >= beta)    return ttScore;
                if (tte->bound == BOUND_UPPER && ttScore <= alpha)   return ttScore;
            }
        }

        int standPat = evaluator_.evaluate(pos);
        int originalAlpha = alpha;

        if (standPat >= beta) {
            tt_.store(key, standPat, Core::Move::none(), 0, BOUND_LOWER, ply);
            return standPat;
        }
        if (standPat > alpha) alpha = standPat;

        // Delta Pruning
        const int DELTA = 975;
        if (standPat + DELTA < alpha) {
            tt_.store(key, alpha, Core::Move::none(), 0, BOUND_UPPER, ply);
            return alpha;
        }

        Core::MoveList moves;
        Core::generate_legal_moves(pos, moves);
        
        Core::MoveList captures;
        for (int i = 0; i < moves.size(); ++i) {
            if (moves[i].is_capture() || moves[i].is_promotion()) {
                captures.push_back(moves[i]);
            }
        }
        
        ordering_.sort_moves(pos, captures, ttMove, ply);

        Core::Move bestMove = Core::Move::none();
        int bestScore = standPat;

        for (int i = 0; i < captures.size(); ++i) {
            const Core::Move move = captures[i];

            Core::UndoInfo undo{};
            pos.make_move(move, undo);
            int score = -quiescence(pos, -beta, -alpha, ply + 1, limits, callbacks);
            pos.unmake_move(move, undo);

            if (should_stop(limits, callbacks)) return 0;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
            if (score >= beta) {
                tt_.store(key, score, move, 0, BOUND_LOWER, ply);
                return score;
            }
            if (score > alpha) alpha = score;
        }

        TTBound bound = (bestScore <= originalAlpha) ? BOUND_UPPER : BOUND_EXACT;
        tt_.store(key, bestScore, bestMove, 0, bound, ply);

        return bestScore;
    }

    int EngineSearch::evaluate(const Core::Position& pos) {
        return evaluator_.evaluate(pos);
    }

    int EngineSearch::negamax(
        Core::Position& pos,
        int depth,
        int alpha,
        int beta,
        int ply,
        const Limits& limits,
        const Callbacks& callbacks
    ) {

        if ((nodes_ & 2047) == 0 && should_stop(limits, callbacks)) return evaluator_.evaluate(pos);

        if (depth <= 0) return quiescence(pos, alpha, beta, ply, limits, callbacks);

        // Check for draws (repetition or 50-move rule)
        if (ply > 0 && (pos.is_repetition() || pos.halfmove_clock() >= 100)) return 0;

        nodes_++;

        const int originalAlpha = alpha;
        const bool isPVNode = (beta - alpha > 1);
        
        const bool inCheck = pos.in_check();

        uint64_t key = pos.hash();
        const TTEntry* tte = tt_.probe(key);
        Core::Move ttMove = Core::Move::none();

        n_probes++;
        if (tte != nullptr) {
            n_hits++;
        }

        if (tte != nullptr) {
            ttMove = tte->move;

            if (!isPVNode && tte->depth >= depth) {
                int ttScore = TranspositionTable::scoreFromTT(tte->score, ply);

                if (tte->bound == BOUND_EXACT) { return ttScore; }
                if (tte->bound == BOUND_LOWER && ttScore >= beta) { return ttScore; }
                if (tte->bound == BOUND_UPPER && ttScore <= alpha) { return ttScore; }
            }
        }

        // Null Move Pruning (NMP)
        // If we are not in check and our static position is strong (>= beta),
        // we can try giving the opponent a free move. If they still can't raise
        // their score above alpha, our position is likely so good we don't need to search deeper.
        if (depth >= 3 && !inCheck && ply > 0) {
            int staticEval = evaluator_.evaluate(pos);
            if (staticEval >= beta) {
                Core::UndoInfo undo;
                pos.make_null_move(undo);
                int score = -negamax(pos, depth - 1 - 2, -beta, -beta + 1, ply + 1, limits, callbacks);
                pos.unmake_null_move(undo);

                if (should_stop(limits, callbacks)) return 0;
                if (score >= beta) return beta;
            }
        }

        Core::MoveList moves;
        Core::generate_legal_moves(pos, moves);
        if (moves.size() == 0) {
            return inCheck ? (-MATE_SCORE + ply) : 0;
        }

        ordering_.sort_moves(pos, moves, ttMove, ply);

        const Core::Color us = pos.side_to_move();
        Core::Move bestMove = moves[0];
        int bestScore = -INF_SCORE;
        bool foundPv = false;
        for (int i = 0; i < moves.size(); ++i) {
            const Core::Move move = moves[i];
            Core::UndoInfo undo{};
            pos.make_move(move, undo);

            int score;
            if (i == 0) {
                score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, limits, callbacks);
            } else {
                // Compute LMR reduction for late, quiet, non-check moves
                int reduction = 0;
                if (depth >= 3 && i >= 3 && !move.is_capture() && !move.is_promotion() && !inCheck) {
                    reduction = 1;
                    if (depth >= 6) reduction = 2;
                }

                // Null-window search at reduced depth
                score = -negamax(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, limits, callbacks);

                // Re-search at full depth if reduced search beat alpha, or if PVS null window failed high
                if (score > alpha) {
                    score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, limits, callbacks);
                }
            }

            pos.unmake_move(move, undo);

            if (should_stop(limits, callbacks)) return score;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha) {
                alpha = score;
                foundPv = true; // We found a new best move, so we have a PV
                if (!move.is_capture()) ordering_.update_history(us, move, depth);
            }

            if (alpha >= beta) {
                if (!move.is_capture()) ordering_.update_killers(ply, move);
                break; // Beta Cutoff
            }
        }
        TTBound bound;
        if (bestScore <= originalAlpha) {
            bound = BOUND_UPPER;
        } else if (bestScore >= beta) {
            bound = BOUND_LOWER;
        } else {
            bound = BOUND_EXACT;
        }

        tt_.store(key, bestScore, bestMove, depth, bound, ply);

        return bestScore;
    }

    int EngineSearch::search_root(
        Core::Position& root,
        Core::MoveList& rootMoves,
        int depth,
        Core::Move& bestMove,
        const Limits& limits,
        const Callbacks& callbacks
    ) {
        const TTEntry* tte = tt_.probe(root.hash());
        Core::Move ttMove = tte ? tte->move : Core::Move::none();

        ordering_.sort_moves(root, rootMoves, ttMove, 0);

        int alpha = -INF_SCORE;
        int beta = INF_SCORE;
        int bestScore = -INF_SCORE;
        bestMove = rootMoves[0];

        const Core::Color us = root.side_to_move();

        for (int i = 0; i < rootMoves.size(); ++i) {
            const Core::Move move = rootMoves[i];

            Core::UndoInfo undo{};
            root.make_move(move, undo);

            int score;
            if (i == 0) {
                score = -negamax(root, depth - 1, -beta, -alpha, 1, limits, callbacks);
            } else {
                // PVS at root
                score = -negamax(root, depth - 1, -alpha - 1, -alpha, 1, limits, callbacks);
                if (score > alpha) {
                    score = -negamax(root, depth - 1, -beta, -alpha, 1, limits, callbacks);
                }
            }

            root.unmake_move(move, undo);

            if (should_stop(limits, callbacks)) break;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha) {
                alpha = score;
                if (!move.is_capture()) ordering_.update_history(us, move, depth);
            }
        }

        if (bestScore > -INF_SCORE) {
            tt_.store(root.hash(), bestScore, bestMove, depth, BOUND_EXACT, 0);
        }

        return bestScore;
    }

    Result EngineSearch::search(Core::Position root, const Limits& limits, const Callbacks& callbacks) {
        Result out{};
        nodes_ = 0;
        started_ = Clock::now();

        ordering_.clear();
        ordering_.age_history();

        Core::MoveList rootMoves;
        Core::generate_legal_moves(root, rootMoves);
        if (rootMoves.size() == 0) {
            out.bestMove = Core::Move::none();
            out.scoreCp = root.in_check() ? -MATE_SCORE : 0;
            out.nodes = nodes_;
            return out;
        }

        out.bestMove = rootMoves[0];

        for (int depth = 1; depth <= limits.maxDepth; ++depth) {
            if (should_stop(limits, callbacks)) break;

            Core::Move depthBest = out.bestMove;
            q_probes = 0;
            q_hits = 0;
            n_probes = 0;
            n_hits = 0;
            const int score = search_root(root, rootMoves, depth, depthBest, limits, callbacks);

            if (should_stop(limits, callbacks)) break;

            out.bestMove = depthBest;
            out.scoreCp = score;
            out.completedDepth = depth;
            out.nodes = nodes_;

            if (callbacks.onInfo) {
                const int elapsedMs = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started_).count()
                );
                const double qsearchTtHitRate = q_probes > 0 ? (100.0 * q_hits / q_probes) : 0.0;
                const double negamaxTtHitRate = n_probes > 0 ? (100.0 * n_hits / n_probes) : 0.0;
                callbacks.onInfo(depth, score, depthBest, nodes_, elapsedMs, qsearchTtHitRate, negamaxTtHitRate);
            }
        }

        // ...
        // If even depth 1 didn't finish (very tight deadline), fallback to static eval or first move
        if (out.completedDepth == 0) {
             // Basic fallback: just return the first move and static eval
             out.scoreCp = evaluator_.evaluate(root);
             out.nodes = nodes_;
        }

        return out;
    }
}

#include "search.h"

#include <algorithm>

namespace Search {
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

    int EngineSearch::negamax(
        Core::Position& pos,
        int depth,
        int alpha,
        int beta,
        int ply,
        const Limits& limits,
        const Callbacks& callbacks
    ) {
        if (should_stop(limits, callbacks)) return evaluator_.evaluate(pos);
        if (ply >= MAX_PLY - 1) return evaluator_.evaluate(pos);

        nodes_++;

        if (depth <= 0) return evaluator_.evaluate(pos);
        if (pos.is_repetition() || pos.halfmove_clock() >= 100) return 0;

        const int alphaOrig = alpha;
        TTEntry tte{};
        Core::Move ttMove = Core::Move::none();
        if (tt_.probe(pos.hash(), tte)) {
            ttMove = Core::Move(static_cast<uint16_t>(tte.bestMove & 0xFFFFu));
            if (tte.depth >= depth) {
                const int ttScore = tte.score;
                if (tte.flag == TTFlag::EXACT) return ttScore;
                if (tte.flag == TTFlag::LOWER) alpha = std::max(alpha, ttScore);
                else if (tte.flag == TTFlag::UPPER) beta = std::min(beta, ttScore);
                if (alpha >= beta) return ttScore;
            }
        }

        Core::MoveList moves;
        Core::generate_legal_moves(pos, moves);
        if (moves.size() == 0) {
            return pos.in_check() ? (-MATE_SCORE + ply) : 0;
        }

        ordering_.sort_moves(pos, moves, ttMove, ply);

        const Core::Color us = pos.side_to_move();
        Core::Move bestMove = moves[0];
        int bestScore = -INF_SCORE;

        for (int i = 0; i < moves.size(); ++i) {
            const Core::Move move = moves[i];
            Core::UndoInfo undo{};
            pos.make_move(move, undo);
            const int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, limits, callbacks);
            pos.unmake_move(move, undo);

            if (should_stop(limits, callbacks)) return score;

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }

            if (score > alpha) {
                alpha = score;
                if (!move.is_capture()) ordering_.update_history(us, move, depth);
            }

            if (alpha >= beta) {
                if (!move.is_capture()) ordering_.update_killers(ply, move);
                break;
            }
        }

        TTFlag flag = TTFlag::EXACT;
        if (bestScore <= alphaOrig) flag = TTFlag::UPPER;
        else if (bestScore >= beta) flag = TTFlag::LOWER;
        tt_.store(pos.hash(), depth, bestScore, flag, bestMove);

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
        TTEntry tte{};
        Core::Move ttMove = Core::Move::none();
        if (tt_.probe(root.hash(), tte)) ttMove = Core::Move(static_cast<uint16_t>(tte.bestMove & 0xFFFFu));

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
            const int score = -negamax(root, depth - 1, -beta, -alpha, 1, limits, callbacks);
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
            tt_.store(root.hash(), depth, bestScore, TTFlag::EXACT, bestMove);
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
                callbacks.onInfo(depth, score, depthBest, nodes_, elapsedMs);
            }
        }

        if (out.completedDepth == 0) {
            out.scoreCp = evaluator_.evaluate(root);
            out.nodes = nodes_;
        }

        return out;
    }
}

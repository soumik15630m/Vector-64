#include "uci.h"

#include "../cores/attacks.h"
#include "../cores/bitboard.h"
#include "../cores/movegen.h"
#include "../cores/position.h"
#include "../cores/zobrist.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace UCI {
    namespace {
        using Clock = std::chrono::steady_clock;
        using Ms = std::chrono::milliseconds;

        constexpr int INF_SCORE = 1000000;
        constexpr int MATE_SCORE = 900000;
        constexpr int MAX_DEPTH = 64;
        constexpr const char* STANDARD_STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        struct GoParams {
            bool ponder = false;
            bool infinite = false;
            int depth = -1;
            int movetimeMs = -1;
            int wtimeMs = -1;
            int btimeMs = -1;
            int wincMs = 0;
            int bincMs = 0;
            int movesToGo = 0;
            uint64_t nodes = 0;
        };

        struct SearchLimits {
            bool infinite = false;
            int maxDepth = MAX_DEPTH;
            uint64_t maxNodes = 0;
            bool hasDeadline = false;
            Clock::time_point deadline;
        };

        class ThreadPool {
        public:
            explicit ThreadPool(size_t count) {
                resize(count);
            }

            ~ThreadPool() {
                stop();
            }

            void resize(size_t count) {
                stop();
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    stopping_ = false;
                }
                workers_.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    workers_.emplace_back([this]() { worker_loop(); });
                }
            }

            template <typename Fn>
            auto enqueue(Fn&& fn) -> std::future<typename std::invoke_result_t<Fn&>> {
                using Ret = typename std::invoke_result_t<Fn&>;
                auto task = std::make_shared<std::packaged_task<Ret()>>(std::forward<Fn>(fn));
                std::future<Ret> result = task->get_future();
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    tasks_.emplace([task]() { (*task)(); });
                }
                cv_.notify_one();
                return result;
            }

        private:
            void stop() {
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    stopping_ = true;
                }
                cv_.notify_all();
                for (std::thread& t : workers_) {
                    if (t.joinable()) t.join();
                }
                workers_.clear();
                std::queue<std::function<void()>> empty;
                std::swap(tasks_, empty);
            }

            void worker_loop() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mu_);
                        cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
                        if (stopping_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            }

            std::mutex mu_;
            std::condition_variable cv_;
            bool stopping_ = false;
            std::vector<std::thread> workers_;
            std::queue<std::function<void()>> tasks_;
        };

        std::string to_lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return s;
        }

        std::vector<std::string> split_ws(const std::string& line) {
            std::vector<std::string> tokens;
            std::istringstream iss(line);
            for (std::string tok; iss >> tok;) tokens.push_back(tok);
            return tokens;
        }

        int piece_value(Core::PieceType pt) {
            switch (pt) {
                case Core::PAWN: return 100;
                case Core::KNIGHT: return 320;
                case Core::BISHOP: return 330;
                case Core::ROOK: return 500;
                case Core::QUEEN: return 900;
                case Core::KING: return 0;
                default: return 0;
            }
        }

        int evaluate(const Core::Position& pos) {
            int white = 0;
            int black = 0;
            for (int pt = Core::PAWN; pt <= Core::KING; ++pt) {
                int v = piece_value(static_cast<Core::PieceType>(pt));
                white += Core::popcount(pos.pieces(static_cast<Core::PieceType>(pt), Core::WHITE)) * v;
                black += Core::popcount(pos.pieces(static_cast<Core::PieceType>(pt), Core::BLACK)) * v;
            }
            int score = white - black;
            return pos.side_to_move() == Core::WHITE ? score : -score;
        }

        bool parse_square(const std::string& s, Core::Square& out) {
            if (s.size() != 2) return false;
            char f = static_cast<char>(std::tolower(static_cast<unsigned char>(s[0])));
            char r = s[1];
            if (f < 'a' || f > 'h') return false;
            if (r < '1' || r > '8') return false;
            out = Core::make_square(static_cast<Core::GenFile>(f - 'a'), static_cast<Core::GenRank>(r - '1'));
            return true;
        }

        char promo_to_char(Core::PieceType pt) {
            switch (pt) {
                case Core::KNIGHT: return 'n';
                case Core::BISHOP: return 'b';
                case Core::ROOK: return 'r';
                case Core::QUEEN: return 'q';
                default: return 'q';
            }
        }

        std::string move_to_uci(Core::Move m) {
            if (!m.is_ok()) return "0000";
            char out[6] = {0, 0, 0, 0, 0, 0};
            out[0] = static_cast<char>('a' + Core::file_of(m.from_sq()));
            out[1] = static_cast<char>('1' + Core::rank_of(m.from_sq()));
            out[2] = static_cast<char>('a' + Core::file_of(m.to_sq()));
            out[3] = static_cast<char>('1' + Core::rank_of(m.to_sq()));
            if (m.is_promotion()) {
                out[4] = promo_to_char(m.promotion_type());
                return std::string(out, out + 5);
            }
            return std::string(out, out + 4);
        }

        bool move_matches_uci(Core::Move m, const std::string& uci) {
            if (uci.size() < 4 || uci.size() > 5) return false;
            Core::Square from = Core::SQ_NONE;
            Core::Square to = Core::SQ_NONE;
            if (!parse_square(uci.substr(0, 2), from)) return false;
            if (!parse_square(uci.substr(2, 2), to)) return false;
            if (m.from_sq() != from || m.to_sq() != to) return false;
            if (uci.size() == 5) {
                if (!m.is_promotion()) return false;
                char p = static_cast<char>(std::tolower(static_cast<unsigned char>(uci[4])));
                return promo_to_char(m.promotion_type()) == p;
            }
            return !m.is_promotion();
        }

        class EngineUci {
        public:
            EngineUci() {
                const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
                threads_ = static_cast<int>(hc);
                pool_ = std::make_unique<ThreadPool>(static_cast<size_t>(threads_));
                position_.setFromFEN(STANDARD_STARTPOS_FEN);
            }

            ~EngineUci() {
                stop_and_join(true);
            }

            int loop() {
                for (std::string line; std::getline(std::cin, line);) {
                    if (!handle_command(line)) return 0;
                }
                stop_and_join(true);
                return 0;
            }

        private:
            bool handle_command(const std::string& line) {
                std::vector<std::string> tokens = split_ws(line);
                if (tokens.empty()) return true;

                std::string cmd = to_lower(tokens[0]);
                if (cmd == "uci") {
                    handle_uci();
                    return true;
                }
                if (cmd == "isready") {
                    emit("readyok");
                    return true;
                }
                if (cmd == "setoption") {
                    handle_setoption(line);
                    return true;
                }
                if (cmd == "ucinewgame") {
                    stop_and_join(true);
                    std::lock_guard<std::mutex> lock(positionMu_);
                    position_.setFromFEN(STANDARD_STARTPOS_FEN);
                    return true;
                }
                if (cmd == "position") {
                    handle_position(tokens);
                    return true;
                }
                if (cmd == "go") {
                    handle_go(tokens);
                    return true;
                }
                if (cmd == "stop") {
                    stopRequested_.store(true, std::memory_order_relaxed);
                    return true;
                }
                if (cmd == "ponderhit") {
                    stopRequested_.store(false, std::memory_order_relaxed);
                    return true;
                }
                if (cmd == "quit") {
                    stop_and_join(true);
                    return false;
                }
                return true;
            }

            void handle_uci() {
                emit("id name STK-Vector-64");
                emit("id author Soumik");
                emit("option name Threads type spin default " + std::to_string(threads_) + " min 1 max 64");
                emit("option name Hash type spin default " + std::to_string(hashMb_) + " min 1 max 4096");
                emit("option name Move Overhead type spin default " + std::to_string(moveOverheadMs_) + " min 0 max 500");
                emit("option name Ponder type check default " + std::string(ponder_ ? "true" : "false"));
                emit("uciok");
            }

            void handle_setoption(const std::string& line) {
                std::string lower = to_lower(line);
                size_t namePos = lower.find("name ");
                if (namePos == std::string::npos) return;
                size_t valuePos = lower.find(" value ");

                std::string name;
                std::string value;
                if (valuePos == std::string::npos) {
                    name = line.substr(namePos + 5);
                } else {
                    name = line.substr(namePos + 5, valuePos - (namePos + 5));
                    value = line.substr(valuePos + 7);
                }

                name = to_lower(name);
                while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
                while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front()))) name.erase(name.begin());

                if (name == "threads") {
                    int newThreads = threads_;
                    try {
                        newThreads = std::stoi(value);
                    } catch (...) {
                        return;
                    }
                    newThreads = std::max(1, std::min(64, newThreads));
                    stop_and_join(true);
                    threads_ = newThreads;
                    pool_ = std::make_unique<ThreadPool>(static_cast<size_t>(threads_));
                    return;
                }

                if (name == "hash") {
                    int newHash = hashMb_;
                    try {
                        newHash = std::stoi(value);
                    } catch (...) {
                        return;
                    }
                    hashMb_ = std::max(1, std::min(4096, newHash));
                    return;
                }

                if (name == "move overhead") {
                    int newOverhead = moveOverheadMs_;
                    try {
                        newOverhead = std::stoi(value);
                    } catch (...) {
                        return;
                    }
                    moveOverheadMs_ = std::max(0, std::min(500, newOverhead));
                    return;
                }

                if (name == "ponder") {
                    ponder_ = to_lower(value) == "true";
                }
            }

            void handle_position(const std::vector<std::string>& tokens) {
                stop_and_join(true);

                if (tokens.size() < 2) return;
                Core::Position next;
                size_t i = 1;

                if (tokens[i] == "startpos") {
                    next.setFromFEN(STANDARD_STARTPOS_FEN);
                    ++i;
                } else if (tokens[i] == "fen") {
                    ++i;
                    std::string fen;
                    int parts = 0;
                    while (i < tokens.size() && tokens[i] != "moves" && parts < 6) {
                        if (!fen.empty()) fen += " ";
                        fen += tokens[i++];
                        ++parts;
                    }
                    if (parts < 4 || !next.setFromFEN(fen)) {
                        emit("info string invalid fen in position command");
                        return;
                    }
                } else {
                    emit("info string invalid position command");
                    return;
                }

                if (i < tokens.size() && tokens[i] == "moves") ++i;
                for (; i < tokens.size(); ++i) {
                    Core::MoveList legal;
                    Core::generate_legal_moves(next, legal);
                    bool found = false;
                    Core::Move chosen;
                    for (int k = 0; k < legal.size(); ++k) {
                        if (move_matches_uci(legal[k], tokens[i])) {
                            chosen = legal[k];
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        emit("info string illegal move in position command: " + tokens[i]);
                        return;
                    }
                    Core::UndoInfo ui;
                    next.make_move(chosen, ui);
                }

                std::lock_guard<std::mutex> lock(positionMu_);
                position_ = next;
            }

            void handle_go(const std::vector<std::string>& tokens) {
                GoParams params;
                for (size_t i = 1; i < tokens.size(); ++i) {
                    const std::string key = tokens[i];
                    if (key == "ponder") params.ponder = true;
                    else if (key == "infinite") params.infinite = true;
                    else if (key == "depth" && i + 1 < tokens.size()) params.depth = std::max(1, std::stoi(tokens[++i]));
                    else if (key == "movetime" && i + 1 < tokens.size()) params.movetimeMs = std::max(1, std::stoi(tokens[++i]));
                    else if (key == "wtime" && i + 1 < tokens.size()) params.wtimeMs = std::max(0, std::stoi(tokens[++i]));
                    else if (key == "btime" && i + 1 < tokens.size()) params.btimeMs = std::max(0, std::stoi(tokens[++i]));
                    else if (key == "winc" && i + 1 < tokens.size()) params.wincMs = std::max(0, std::stoi(tokens[++i]));
                    else if (key == "binc" && i + 1 < tokens.size()) params.bincMs = std::max(0, std::stoi(tokens[++i]));
                    else if (key == "movestogo" && i + 1 < tokens.size()) params.movesToGo = std::max(0, std::stoi(tokens[++i]));
                    else if (key == "nodes" && i + 1 < tokens.size()) params.nodes = static_cast<uint64_t>(std::stoull(tokens[++i]));
                }
                start_search(params);
            }

            void start_search(const GoParams& params) {
                std::lock_guard<std::mutex> lock(searchMu_);

                const uint64_t searchId = activeSearchId_.fetch_add(1, std::memory_order_relaxed) + 1;
                if (searchThread_.joinable()) {
                    stopRequested_.store(true, std::memory_order_relaxed);
                    searchThread_.join();
                }

                stopRequested_.store(false, std::memory_order_relaxed);
                nodesSearched_.store(0, std::memory_order_relaxed);
                searchThread_ = std::thread([this, searchId, params]() { search_worker(searchId, params); });
            }

            void stop_and_join(bool suppressOutput) {
                std::lock_guard<std::mutex> lock(searchMu_);
                if (!searchThread_.joinable()) return;
                if (suppressOutput) activeSearchId_.fetch_add(1, std::memory_order_relaxed);
                stopRequested_.store(true, std::memory_order_relaxed);
                searchThread_.join();
                stopRequested_.store(false, std::memory_order_relaxed);
            }

            bool should_stop(uint64_t searchId, const SearchLimits& limits) {
                if (searchId != activeSearchId_.load(std::memory_order_relaxed)) return true;
                if (stopRequested_.load(std::memory_order_relaxed)) return true;
                if (limits.maxNodes > 0 && nodesSearched_.load(std::memory_order_relaxed) >= limits.maxNodes) {
                    stopRequested_.store(true, std::memory_order_relaxed);
                    return true;
                }
                if (limits.hasDeadline && Clock::now() >= limits.deadline) {
                    stopRequested_.store(true, std::memory_order_relaxed);
                    return true;
                }
                return false;
            }

            SearchLimits compute_limits(const Core::Position& pos, const GoParams& params) const {
                SearchLimits limits;
                limits.infinite = params.infinite || params.ponder;
                limits.maxDepth = params.depth > 0 ? std::min(params.depth, MAX_DEPTH) : MAX_DEPTH;
                limits.maxNodes = params.nodes;

                if (params.movetimeMs > 0) {
                    const int t = std::max(1, params.movetimeMs - moveOverheadMs_);
                    limits.hasDeadline = true;
                    limits.deadline = Clock::now() + Ms(t);
                    return limits;
                }

                if (!limits.infinite) {
                    int remain = pos.side_to_move() == Core::WHITE ? params.wtimeMs : params.btimeMs;
                    int inc = pos.side_to_move() == Core::WHITE ? params.wincMs : params.bincMs;
                    if (remain > 0) {
                        int mtg = params.movesToGo > 0 ? params.movesToGo : 30;
                        int slice = remain / std::max(1, mtg);
                        int budget = slice + static_cast<int>(inc * 0.7);
                        budget = std::max(1, budget - moveOverheadMs_);
                        budget = std::min(budget, std::max(1, remain - moveOverheadMs_));
                        limits.hasDeadline = true;
                        limits.deadline = Clock::now() + Ms(budget);
                    }
                }
                return limits;
            }

            int negamax(Core::Position& pos, int depth, int alpha, int beta, int ply, uint64_t searchId, const SearchLimits& limits) {
                if (should_stop(searchId, limits)) return evaluate(pos);
                nodesSearched_.fetch_add(1, std::memory_order_relaxed);
                if (depth <= 0) return evaluate(pos);

                Core::MoveList moves;
                Core::generate_legal_moves(pos, moves);

                if (moves.size() == 0) {
                    if (pos.in_check()) return -MATE_SCORE + ply;
                    return 0;
                }

                int best = -INF_SCORE;
                for (int i = 0; i < moves.size(); ++i) {
                    Core::UndoInfo ui;
                    pos.make_move(moves[i], ui);
                    int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1, searchId, limits);
                    pos.unmake_move(moves[i], ui);

                    if (should_stop(searchId, limits)) return score;
                    if (score > best) best = score;
                    if (best > alpha) alpha = best;
                    if (alpha >= beta) break;
                }
                return best;
            }

            int search_root_parallel(const Core::Position& root, const Core::MoveList& rootMoves, int depth, Core::Move& bestMove, uint64_t searchId, const SearchLimits& limits) {
                std::vector<std::future<int>> futures;
                futures.reserve(static_cast<size_t>(rootMoves.size()));

                for (int i = 0; i < rootMoves.size(); ++i) {
                    Core::Move m = rootMoves[i];
                    futures.push_back(pool_->enqueue([this, root, m, depth, searchId, limits]() mutable {
                        Core::Position child = root;
                        Core::UndoInfo ui;
                        child.make_move(m, ui);
                        return -negamax(child, depth - 1, -INF_SCORE, INF_SCORE, 1, searchId, limits);
                    }));
                }

                int bestScore = -INF_SCORE;
                bestMove = rootMoves[0];
                for (int i = 0; i < rootMoves.size(); ++i) {
                    int score = futures[static_cast<size_t>(i)].get();
                    if (score > bestScore) {
                        bestScore = score;
                        bestMove = rootMoves[i];
                    }
                }
                return bestScore;
            }

            int search_root_serial(Core::Position& root, const Core::MoveList& rootMoves, int depth, Core::Move& bestMove, uint64_t searchId, const SearchLimits& limits) {
                int bestScore = -INF_SCORE;
                bestMove = rootMoves[0];
                for (int i = 0; i < rootMoves.size(); ++i) {
                    Core::UndoInfo ui;
                    root.make_move(rootMoves[i], ui);
                    int score = -negamax(root, depth - 1, -INF_SCORE, INF_SCORE, 1, searchId, limits);
                    root.unmake_move(rootMoves[i], ui);
                    if (should_stop(searchId, limits)) break;
                    if (score > bestScore) {
                        bestScore = score;
                        bestMove = rootMoves[i];
                    }
                }
                return bestScore;
            }

            void emit_info(int depth, int scoreCp, const Core::Move& pvMove, Clock::time_point started) {
                const auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - started).count();
                const uint64_t nodes = nodesSearched_.load(std::memory_order_relaxed);
                const uint64_t nps = elapsed > 0 ? static_cast<uint64_t>((nodes * 1000ULL) / static_cast<uint64_t>(elapsed)) : 0ULL;

                std::ostringstream os;
                os << "info depth " << depth
                   << " score cp " << scoreCp
                   << " nodes " << nodes
                   << " nps " << nps
                   << " time " << elapsed
                   << " pv " << move_to_uci(pvMove);
                emit(os.str());
            }

            void emit_bestmove(uint64_t searchId, Core::Move bestMove) {
                if (searchId != activeSearchId_.load(std::memory_order_relaxed)) return;
                emit("bestmove " + move_to_uci(bestMove));
            }

            void search_worker(uint64_t searchId, GoParams params) {
                Core::Position root;
                {
                    std::lock_guard<std::mutex> lock(positionMu_);
                    root = position_;
                }

                SearchLimits limits = compute_limits(root, params);
                Core::MoveList rootMoves;
                Core::generate_legal_moves(root, rootMoves);

                if (rootMoves.size() == 0) {
                    emit_bestmove(searchId, Core::Move::none());
                    return;
                }

                Core::Move bestMove = rootMoves[0];
                Clock::time_point started = Clock::now();

                for (int depth = 1; depth <= limits.maxDepth; ++depth) {
                    if (should_stop(searchId, limits)) break;

                    Core::Move depthBest = bestMove;
                    int score = -INF_SCORE;
                    if (threads_ > 1 && rootMoves.size() > 1) {
                        score = search_root_parallel(root, rootMoves, depth, depthBest, searchId, limits);
                    } else {
                        score = search_root_serial(root, rootMoves, depth, depthBest, searchId, limits);
                    }

                    if (should_stop(searchId, limits)) break;
                    bestMove = depthBest;
                    emit_info(depth, score, depthBest, started);

                    if (params.depth > 0 && depth >= params.depth) break;
                    if (limits.hasDeadline && Clock::now() + Ms(5) >= limits.deadline) break;
                }

                emit_bestmove(searchId, bestMove);
            }

            void emit(const std::string& line) {
                std::lock_guard<std::mutex> lock(ioMu_);
                std::cout << line << '\n' << std::flush;
            }

            std::mutex ioMu_;
            std::mutex positionMu_;
            std::mutex searchMu_;

            Core::Position position_;

            std::atomic<bool> stopRequested_{false};
            std::atomic<uint64_t> activeSearchId_{0};
            std::atomic<uint64_t> nodesSearched_{0};

            std::thread searchThread_;

            int threads_ = 1;
            int hashMb_ = 64;
            int moveOverheadMs_ = 30;
            bool ponder_ = false;
            std::unique_ptr<ThreadPool> pool_;
        };
    }

    int run() {
        Core::Attacks::init();
        Core::Zobrist::init();

        EngineUci app;
        return app.loop();
    }
}

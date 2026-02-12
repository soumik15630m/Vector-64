#include "uci.h"

#include "../cores/attacks.h"
#include "../cores/movegen.h"
#include "../cores/position.h"
#include "../cores/zobrist.h"
#include "../search/search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace UCI {
    namespace {
        using Clock = std::chrono::steady_clock;
        using Ms = std::chrono::milliseconds;

        constexpr int MAX_DEPTH = 64;
        constexpr const char* STANDARD_STARTPOS_FEN =
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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
            int maxDepth = MAX_DEPTH;
            uint64_t maxNodes = 0;
            bool hasDeadline = false;
            Clock::time_point deadline{};
        };

        std::string to_lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return s;
        }

        std::string trim_copy(std::string s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
            return s;
        }

        std::vector<std::string> split_ws(const std::string& line) {
            std::vector<std::string> tokens;
            std::istringstream iss(line);
            for (std::string tok; iss >> tok;) tokens.push_back(tok);
            return tokens;
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
                const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(uci[4])));
                return promo_to_char(m.promotion_type()) == p;
            }
            return !m.is_promotion();
        }

        bool parse_int(const std::string& token, int& out) {
            try {
                size_t consumed = 0;
                const int value = std::stoi(token, &consumed);
                if (consumed != token.size()) return false;
                out = value;
                return true;
            } catch (...) {
                return false;
            }
        }

        bool parse_u64(const std::string& token, uint64_t& out) {
            try {
                size_t consumed = 0;
                const uint64_t value = std::stoull(token, &consumed);
                if (consumed != token.size()) return false;
                out = value;
                return true;
            } catch (...) {
                return false;
            }
        }

        class EngineUci {
        public:
            EngineUci() {
                const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
                threads_ = static_cast<int>(hc);
                position_.setFromFEN(STANDARD_STARTPOS_FEN);
                search_.set_hash_mb(static_cast<size_t>(hashMb_));
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

                const std::string cmd = to_lower(tokens[0]);
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
                    search_.clear();
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
                emit("option name EvalFile type string default <empty>");
                emit("uciok");
            }

            void handle_setoption(const std::string& line) {
                const std::string lower = to_lower(line);
                const size_t namePos = lower.find("name ");
                if (namePos == std::string::npos) return;
                const size_t valuePos = lower.find(" value ");

                std::string name;
                std::string value;
                if (valuePos == std::string::npos) {
                    name = line.substr(namePos + 5);
                } else {
                    name = line.substr(namePos + 5, valuePos - (namePos + 5));
                    value = line.substr(valuePos + 7);
                }

                name = to_lower(trim_copy(name));
                value = trim_copy(value);

                if (name == "threads") {
                    int parsed = threads_;
                    if (!parse_int(value, parsed)) return;
                    threads_ = std::clamp(parsed, 1, 64);
                    return;
                }

                if (name == "hash") {
                    int parsed = hashMb_;
                    if (!parse_int(value, parsed)) return;
                    hashMb_ = std::clamp(parsed, 1, 4096);
                    search_.set_hash_mb(static_cast<size_t>(hashMb_));
                    return;
                }

                if (name == "move overhead") {
                    int parsed = moveOverheadMs_;
                    if (!parse_int(value, parsed)) return;
                    moveOverheadMs_ = std::clamp(parsed, 0, 500);
                    return;
                }

                if (name == "ponder") {
                    ponder_ = to_lower(value) == "true";
                    return;
                }

                if (name == "evalfile") {
                    std::string path = value;
                    if (path.size() >= 2 && path.front() == '"' && path.back() == '"') {
                        path = path.substr(1, path.size() - 2);
                    }

                    if (path.empty() || to_lower(path) == "<empty>") {
                        emit("info string EvalFile ignored: empty path");
                        return;
                    }

                    stop_and_join(true);
                    if (search_.load_nnue(path)) {
                        emit("info string EvalFile loaded: " + path);
                    } else {
                        emit("info string EvalFile load failed: " + path);
                    }
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

                    Core::UndoInfo undo{};
                    next.make_move(chosen, undo);
                }

                std::lock_guard<std::mutex> lock(positionMu_);
                position_ = next;
            }

            void handle_go(const std::vector<std::string>& tokens) {
                GoParams params;
                for (size_t i = 1; i < tokens.size(); ++i) {
                    const std::string& key = tokens[i];
                    int parsed = 0;
                    uint64_t parsedU64 = 0;

                    if (key == "ponder") params.ponder = true;
                    else if (key == "infinite") params.infinite = true;
                    else if (key == "depth" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.depth = std::max(1, parsed);
                        ++i;
                    } else if (key == "movetime" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.movetimeMs = std::max(1, parsed);
                        ++i;
                    } else if (key == "wtime" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.wtimeMs = std::max(0, parsed);
                        ++i;
                    } else if (key == "btime" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.btimeMs = std::max(0, parsed);
                        ++i;
                    } else if (key == "winc" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.wincMs = std::max(0, parsed);
                        ++i;
                    } else if (key == "binc" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.bincMs = std::max(0, parsed);
                        ++i;
                    } else if (key == "movestogo" && i + 1 < tokens.size() && parse_int(tokens[i + 1], parsed)) {
                        params.movesToGo = std::max(0, parsed);
                        ++i;
                    } else if (key == "nodes" && i + 1 < tokens.size() && parse_u64(tokens[i + 1], parsedU64)) {
                        params.nodes = parsedU64;
                        ++i;
                    }
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

            SearchLimits compute_limits(const Core::Position& pos, const GoParams& params) const {
                SearchLimits limits;
                limits.maxDepth = params.depth > 0 ? std::min(params.depth, MAX_DEPTH) : MAX_DEPTH;
                limits.maxNodes = params.nodes;

                if (params.movetimeMs > 0) {
                    const int budget = std::max(1, params.movetimeMs - moveOverheadMs_);
                    limits.hasDeadline = true;
                    limits.deadline = Clock::now() + Ms(budget);
                    return limits;
                }

                if (params.infinite || params.ponder) return limits;

                const int remain = pos.side_to_move() == Core::WHITE ? params.wtimeMs : params.btimeMs;
                const int inc = pos.side_to_move() == Core::WHITE ? params.wincMs : params.bincMs;
                if (remain > 0) {
                    const int mtg = params.movesToGo > 0 ? params.movesToGo : 30;
                    const int slice = remain / std::max(1, mtg);
                    int budget = slice + static_cast<int>(inc * 0.7);
                    budget = std::max(1, budget - moveOverheadMs_);
                    budget = std::min(budget, std::max(1, remain - moveOverheadMs_));
                    limits.hasDeadline = true;
                    limits.deadline = Clock::now() + Ms(budget);
                }
                return limits;
            }

            void emit_info(int depth, int scoreCp, const Core::Move& pvMove, uint64_t nodes, int elapsedMs) {
                const uint64_t nps = elapsedMs > 0 ? (nodes * 1000ULL) / static_cast<uint64_t>(elapsedMs) : 0ULL;

                std::ostringstream os;
                os << "info depth " << depth
                   << " score cp " << scoreCp
                   << " nodes " << nodes
                   << " nps " << nps
                   << " time " << elapsedMs
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

                const SearchLimits limits = compute_limits(root, params);

                Search::Limits searchLimits;
                searchLimits.maxDepth = limits.maxDepth;
                searchLimits.maxNodes = limits.maxNodes;
                searchLimits.hasDeadline = limits.hasDeadline;
                searchLimits.deadline = limits.deadline;

                Search::Callbacks callbacks;
                callbacks.shouldStop = [this, searchId]() {
                    return stopRequested_.load(std::memory_order_relaxed) ||
                           searchId != activeSearchId_.load(std::memory_order_relaxed);
                };
                callbacks.onInfo = [this, searchId](int depth, int scoreCp, Core::Move pvMove, uint64_t nodes, int elapsedMs) {
                    if (searchId != activeSearchId_.load(std::memory_order_relaxed)) return;
                    emit_info(depth, scoreCp, pvMove, nodes, elapsedMs);
                };

                const Search::Result result = search_.search(root, searchLimits, callbacks);
                emit_bestmove(searchId, result.bestMove);
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
            std::thread searchThread_;

            int threads_ = 1;
            int hashMb_ = 64;
            int moveOverheadMs_ = 30;
            bool ponder_ = false;
            Search::EngineSearch search_{64};
        };
    }

    int run() {
        Core::Attacks::init();
        Core::Zobrist::init();

        EngineUci app;
        return app.loop();
    }
}

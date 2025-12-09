#!/usr/bin/env python3
"""
wiki_crawler.py

Wikipedia link-path finder with strategies:
 - bfs   : uni-directional BFS (guaranteed shortest)
 - best  : heuristic best-first (fast, not guaranteed)
 - bidi  : bidirectional BFS using prop=linkshere (guaranteed shortest, much faster)

Includes:
 - --verbose tracing
 - --flowchart output (PNG) that highlights the final path
 - in-memory caching of links and linkshere (incoming links)
 - polite sleeps between requests

Usage examples:
  python3 wiki_crawler.py --start "Dog" --target "Albert Einstein" --strategy bidi --verbose --flowchart graph.png
"""

import argparse
import requests
import time
from collections import deque
from heapq import heappush, heappop
import difflib
import networkx as nx
import matplotlib.pyplot as plt
import os

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "WikiCrawlerBot/1.0 (example@example.com) Python/requests"

class WikipediaAPIError(Exception):
    pass

class WikiCrawler:
    def __init__(self, session=None, user_agent=None, sleep_between_requests=0.1, verbose=False):
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or DEFAULT_USER_AGENT
        })
        self.sleep = sleep_between_requests
        self.verbose = verbose

        # Caches
        self.link_cache = {}       # canonical title -> set(of outgoing linked titles)
        self.linkshere_cache = {}  # canonical title -> set(of incoming titles)
        self.title_cache = {}      # input title -> canonical title

        # Graph tracking for visualization
        self.crawl_graph = nx.DiGraph()

    def log(self, *msg):
        if self.verbose:
            print("[verbose]", *msg)

    def _api_get(self, params):
        params = dict(params)
        params.setdefault("format", "json")
        resp = self.session.get(API_ENDPOINT, params=params, timeout=30)
        if resp.status_code != 200:
            raise WikipediaAPIError(f"Bad status {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def resolve_title(self, title):
        if title in self.title_cache:
            self.log("Title cache hit:", title, "->", self.title_cache[title])
            return self.title_cache[title]

        self.log("Resolving title:", title)
        params = {
            "action": "query",
            "titles": title,
            "redirects": 1,
            "formatversion": 2,
            "prop": "info"
        }
        j = self._api_get(params)
        pages = j.get("query", {}).get("pages", [])
        if not pages:
            self.log(f"  → No pages found resolving: {title}")
            return None
        page = pages[0]
        if "missing" in page:
            self.log(f"  → Page missing for: {title}")
            return None
        normalized_title = page.get("title")
        self.title_cache[title] = normalized_title
        self.title_cache[normalized_title] = normalized_title
        self.log(f"  → Resolved '{title}' to canonical '{normalized_title}'")
        return normalized_title

    def random_page_title(self, namespace=0):
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": namespace,
            "rnlimit": 1,
            "formatversion": 2
        }
        j = self._api_get(params)
        entries = j.get("query", {}).get("random", [])
        if not entries:
            raise WikipediaAPIError("Could not fetch random page")
        title = entries[0]["title"]
        self.log("Random page picked:", title)
        return title

    def get_links(self, title):
        """
        Outgoing links from `title` (namespace 0 only). Cached.
        """
        normalized_title = self.resolve_title(title)
        if not normalized_title:
            self.log(f"get_links: cannot resolve title '{title}' -> returning empty set")
            return set()

        if normalized_title in self.link_cache:
            self.log("Link cache hit for:", normalized_title, f"({len(self.link_cache[normalized_title])} links)")
            return self.link_cache[normalized_title]

        self.log("Fetching links for:", normalized_title)
        links = set()
        params = {
            "action": "query",
            "titles": normalized_title,
            "prop": "links",
            "pllimit": "max",
            "formatversion": 2
        }

        while True:
            j = self._api_get(params)
            time.sleep(self.sleep)
            pages = j.get("query", {}).get("pages", [])
            if pages:
                page = pages[0]
                for l in page.get("links", []):
                    if l.get("ns") == 0:
                        links.add(l.get("title"))
            cont = j.get("continue")
            if cont and cont.get("plcontinue"):
                params["plcontinue"] = cont["plcontinue"]
                self.log("  → continuation, plcontinue:", params["plcontinue"])
                continue
            else:
                break

        self.link_cache[normalized_title] = links
        self.log(f"  → {len(links)} links collected for '{normalized_title}'")
        return links

    def get_linkshere(self, title):
        """
        Incoming links to `title` (pages that link to it). Cached.
        Uses prop=linkshere with lhnamespace=0.
        """
        normalized_title = self.resolve_title(title)
        if not normalized_title:
            self.log(f"get_linkshere: cannot resolve title '{title}' -> returning empty set")
            return set()

        if normalized_title in self.linkshere_cache:
            self.log("Linkshere cache hit for:", normalized_title, f"({len(self.linkshere_cache[normalized_title])} incoming)")
            return self.linkshere_cache[normalized_title]

        self.log("Fetching incoming links (linkshere) for:", normalized_title)
        incoming = set()
        params = {
            "action": "query",
            "titles": normalized_title,
            "prop": "linkshere",
            "lhlimit": "max",
            "lhnamespace": 0,
            "formatversion": 2
        }

        while True:
            j = self._api_get(params)
            time.sleep(self.sleep)
            pages = j.get("query", {}).get("pages", [])
            if pages:
                page = pages[0]
                for l in page.get("linkshere", []):
                    # linkshere returns objects with 'title'
                    incoming.add(l.get("title"))
            cont = j.get("continue")
            if cont and cont.get("lhcontinue"):
                params["lhcontinue"] = cont["lhcontinue"]
                self.log("  → continuation, lhcontinue:", params["lhcontinue"])
                continue
            else:
                break

        self.linkshere_cache[normalized_title] = incoming
        self.log(f"  → {len(incoming)} incoming links collected for '{normalized_title}'")
        return incoming

    def search_title(self, query, limit=1):
        self.log("Searching for best match of:", query)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "formatversion": 2
        }
        j = self._api_get(params)
        hits = j.get("query", {}).get("search", [])
        if not hits:
            self.log("  → No search hits for:", query)
            return None
        title = hits[0]["title"]
        self.log("  → Search matched to:", title)
        return title

    # ------------------------------
    # Uni-directional BFS (kept for fallback)
    # ------------------------------
    def find_path_bfs(self, start_title, target_title, max_depth=6, max_visited=100000):
        start = self.resolve_title(start_title)
        if start is None:
            raise ValueError(f"Start page not found: {start_title}")
        target = self.resolve_title(target_title)
        if target is None:
            raise ValueError(f"Target page not found: {target_title}")

        # reset graph
        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)

        self.log("BFS start:", start, "-> target:", target)
        if start == target:
            return [start]

        q = deque()
        q.append((start, [start], 0))
        visited = {start}
        visited_count = 0

        while q:
            current, path, depth = q.popleft()
            visited_count += 1

            if visited_count % 500 == 0:
                self.log(f"Visited {visited_count} pages so far; queue size: {len(q)}")
            if visited_count > max_visited:
                raise RuntimeError("Visited cap exceeded; aborting")

            self.log(f"Visiting: {current} | Depth: {depth} | Path length: {len(path)} | Queue: {len(q)}")
            if depth >= max_depth:
                self.log(f"  → reached max depth for node {current}, skipping expansion")
                continue

            try:
                neighbors = self.get_links(current)
            except Exception as e:
                print(f"[warning] failed to get links for {current}: {e}")
                neighbors = set()

            # record edges for visualization
            for n in neighbors:
                if not self.crawl_graph.has_node(n):
                    self.crawl_graph.add_node(n)
                if not self.crawl_graph.has_edge(current, n):
                    self.crawl_graph.add_edge(current, n)

            self.log(f"Expanding {current}: {len(neighbors)} neighbors")

            if target in neighbors:
                self.log(f"Target '{target}' found as neighbor of '{current}'")
                return path + [target]

            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    q.append((n, path + [n], depth + 1))
                    self.log(f"  → enqueued: {n} (new depth {depth+1})")
                else:
                    self.log(f"  → skipped visited: {n}")

        self.log("BFS complete: no path found within max_depth")
        return None

    # ------------------------------
    # Best-first (heuristic) search (kept)
    # ------------------------------
    def _title_score(self, candidate_title, target_title):
        ratio = difflib.SequenceMatcher(None, candidate_title.lower(), target_title.lower()).ratio()
        score = ratio
        target_tokens = [t for t in target_title.lower().split() if len(t) > 2]
        cand_lower = candidate_title.lower()
        token_bonus = 0.0
        for tkn in target_tokens:
            if tkn in cand_lower:
                token_bonus += 0.25
        score = score + token_bonus
        return score

    def find_path_best_first(self, start_title, target_title, max_depth=6, max_visited=50000, max_branch=50):
        start = self.resolve_title(start_title)
        if start is None:
            raise ValueError(f"Start page not found: {start_title}")
        target = self.resolve_title(target_title)
        if target is None:
            raise ValueError(f"Target page not found: {target_title}")

        # reset graph
        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)

        if start == target:
            return [start]

        uid_counter = 0
        heap = []
        start_score = self._title_score(start, target)
        heappush(heap, (-start_score, 0, uid_counter, start, [start]))
        uid_counter += 1

        visited = set([start])
        visited_count = 0

        while heap:
            neg_score, depth, _, current, path = heappop(heap)
            visited_count += 1
            if visited_count > max_visited:
                raise RuntimeError("Visited cap exceeded; aborting")

            self.log(f"[best-first] Visiting: {current} | depth={depth} | score={-neg_score:.4f} | path_len={len(path)} | heap={len(heap)}")
            if depth >= max_depth:
                self.log(f"  -> reached max depth for {current}, skipping expansion")
                continue

            try:
                neighbors = self.get_links(current)
            except Exception as e:
                print(f"[warning] failed to get links for {current}: {e}")
                neighbors = set()

            # record edges for visualization
            for n in neighbors:
                if not self.crawl_graph.has_node(n):
                    self.crawl_graph.add_node(n)
                if not self.crawl_graph.has_edge(current, n):
                    self.crawl_graph.add_edge(current, n)

            if target in neighbors:
                self.log(f"[best-first] Target '{target}' found as neighbor of '{current}'")
                return path + [target]

            scored = []
            for n in neighbors:
                if n in visited:
                    continue
                sc = self._title_score(n, target)
                scored.append((sc, n))

            if not scored:
                continue

            scored.sort(reverse=True, key=lambda x: x[0])
            top_neighbors = scored[:max_branch]
            self.log(f"  -> expanding {len(top_neighbors)} of {len(scored)} neighbors (top by heuristic)")

            for sc, n in top_neighbors:
                if n not in visited:
                    visited.add(n)
                    uid_counter += 1
                    heappush(heap, (-sc, depth + 1, uid_counter, n, path + [n]))
                    self.log(f"    enqueued {n} (score={sc:.4f})")
                else:
                    self.log(f"    skipped visited {n}")

        self.log("[best-first] no path found within limits")
        return None

    # ------------------------------
    # Bidirectional BFS using linkshere (exact & faster)
    # ------------------------------
    def find_path_bidi(self, start_title, target_title, max_depth=6, max_visited=100000):
        """
        Bidirectional BFS:
         - forward expands outgoing links using prop=links
         - backward expands incoming links using prop=linkshere
        Returns shortest path in clicks if found (list), else None.
        """
        start = self.resolve_title(start_title)
        if start is None:
            raise ValueError(f"Start page not found: {start_title}")
        target = self.resolve_title(target_title)
        if target is None:
            raise ValueError(f"Target page not found: {target_title}")

        if start == target:
            return [start]

        # reset graph
        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)
        self.crawl_graph.add_node(target)

        # Structures for BFS from both sides
        # parent maps for reconstructing path: child -> parent
        parent_fwd = {start: None}
        parent_bwd = {target: None}

        # frontiers as queues of (node, depth)
        q_fwd = deque([(start, 0)])
        q_bwd = deque([(target, 0)])

        # visited sets
        visited_fwd = {start}
        visited_bwd = {target}

        visited_count = 0

        # We'll expand alternately; keep track of depths to respect max_depth overall.
        # Depth in each side counts clicks from respective root. Meeting depth = d_fwd + d_bwd
        while q_fwd and q_bwd:
            # Decide which frontier to expand: the smaller queue (helps balance)
            if len(q_fwd) <= len(q_bwd):
                # expand one level from forward frontier
                current, depth = q_fwd.popleft()
                visited_count += 1
                if visited_count > max_visited:
                    raise RuntimeError("Visited cap exceeded; aborting")

                self.log(f"[bidi][FWD] Visiting: {current} | Depth: {depth} | q_fwd={len(q_fwd)} | q_bwd={len(q_bwd)}")
                if depth >= max_depth:
                    self.log(f"  -> reached max forward depth for {current}, skipping expansion")
                    # continue loop to potentially expand other side
                else:
                    try:
                        neighbors = self.get_links(current)
                    except Exception as e:
                        print(f"[warning] failed to get links for {current}: {e}")
                        neighbors = set()

                    # record edges current -> neighbor (forward)
                    for n in neighbors:
                        if not self.crawl_graph.has_node(n):
                            self.crawl_graph.add_node(n)
                        if not self.crawl_graph.has_edge(current, n):
                            self.crawl_graph.add_edge(current, n)

                    # check intersection quickly
                    inter = neighbors & visited_bwd
                    if inter:
                        meet = next(iter(inter))
                        self.log(f"[bidi] Meeting node found via forward expansion: {meet}")
                        return self._reconstruct_bidi_path(parent_fwd, parent_bwd, meet, start, target)

                    for n in neighbors:
                        if n not in visited_fwd:
                            visited_fwd.add(n)
                            parent_fwd[n] = current
                            q_fwd.append((n, depth + 1))
                            self.log(f"  -> fwd enqueued: {n} (depth {depth+1})")
                        else:
                            self.log(f"  -> fwd skipped visited: {n}")
            else:
                # expand backward frontier (incoming links)
                current, depth = q_bwd.popleft()
                visited_count += 1
                if visited_count > max_visited:
                    raise RuntimeError("Visited cap exceeded; aborting")

                self.log(f"[bidi][BWD] Visiting: {current} | Depth: {depth} | q_fwd={len(q_fwd)} | q_bwd={len(q_bwd)}")
                if depth >= max_depth:
                    self.log(f"  -> reached max backward depth for {current}, skipping expansion")
                else:
                    try:
                        incoming = self.get_linkshere(current)
                    except Exception as e:
                        print(f"[warning] failed to get linkshere for {current}: {e}")
                        incoming = set()

                    # record edges incoming -> current (because those pages link to current)
                    for n in incoming:
                        if not self.crawl_graph.has_node(n):
                            self.crawl_graph.add_node(n)
                        if not self.crawl_graph.has_edge(n, current):
                            self.crawl_graph.add_edge(n, current)

                    # check intersection quickly
                    inter = incoming & visited_fwd
                    if inter:
                        meet = next(iter(inter))
                        self.log(f"[bidi] Meeting node found via backward expansion: {meet}")
                        return self._reconstruct_bidi_path(parent_fwd, parent_bwd, meet, start, target)

                    for n in incoming:
                        if n not in visited_bwd:
                            visited_bwd.add(n)
                            parent_bwd[n] = current
                            q_bwd.append((n, depth + 1))
                            self.log(f"  -> bwd enqueued: {n} (depth {depth+1})")
                        else:
                            self.log(f"  -> bwd skipped visited: {n}")

        self.log("[bidi] No meeting point found within limits")
        return None

    def _reconstruct_bidi_path(self, parent_fwd, parent_bwd, meeting_node, start, target):
        """
        Reconstruct path from start -> meeting_node using parent_fwd,
        and from meeting_node -> target using parent_bwd.
        parent_fwd: child->parent (towards start)
        parent_bwd: child->parent (towards target)
        """
        # build start -> meeting
        path_left = []
        node = meeting_node
        while node is not None:
            path_left.append(node)
            node = parent_fwd.get(node)
        path_left = list(reversed(path_left))  # now start ... meeting

        # build meeting -> target
        path_right = []
        node = parent_bwd.get(meeting_node)
        # parent_bwd maps child -> parent in backward tree (child is closer to target)
        # we want meeting -> ... -> target
        cur = meeting_node
        while cur is not None and cur != target:
            nxt = parent_bwd.get(cur)
            if nxt is None:
                break
            path_right.append(nxt)
            cur = nxt

        full_path = path_left + path_right
        # ensure start and target included
        if full_path[0] != start:
            full_path = [start] + full_path
        if full_path[-1] != target:
            full_path = full_path + [target]
        self.log("[bidi] Reconstructed path:", " -> ".join(full_path))
        return full_path

    # ------------------------------
    # Flowchart generation
    # ------------------------------
    def draw_flowchart(self, output_path, highlight_path=None, max_nodes=500):
        if self.crawl_graph is None or len(self.crawl_graph.nodes) == 0:
            raise RuntimeError("No crawl graph recorded to draw.")

        G = self.crawl_graph.copy()

        # prune if too large
        if len(G.nodes) > max_nodes:
            self.log("Graph has", len(G.nodes), "nodes; pruning for visualization to", max_nodes, "nodes")
            keep = set()
            if highlight_path:
                for node in highlight_path:
                    keep.add(node)
                    keep.update(G.successors(node))
                    keep.update(G.predecessors(node))
            deg_sorted = sorted(G.nodes, key=lambda n: G.out_degree(n)+G.in_degree(n), reverse=True)
            idx = 0
            while len(keep) < max_nodes and idx < len(deg_sorted):
                keep.add(deg_sorted[idx])
                idx += 1
            G = G.subgraph(keep).copy()

        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(G, k=0.5, iterations=100)

        nx.draw_networkx_nodes(G, pos, node_size=200, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrowsize=12, arrowstyle='->', width=1)

        if highlight_path and len(highlight_path) >= 2:
            path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
            existing_path_edges = [e for e in path_edges if G.has_edge(*e)]
            if existing_path_edges:
                nx.draw_networkx_edges(G, pos, edgelist=existing_path_edges, arrowsize=14, arrowstyle='->', width=3)

        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title("Wikipedia crawl graph (start -> target). Highlighted path is thicker.")
        plt.axis('off')

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        self.log("Flowchart saved to:", output_path)

def main():
    parser = argparse.ArgumentParser(description="Find link path between two Wikipedia pages (bfs, best, bidi).")
    parser.add_argument("--start", help="Start page title (quote if contains spaces).")
    parser.add_argument("--target", required=True, help="Target page title (quote if contains spaces).")
    parser.add_argument("--random-start", action="store_true", help="Pick a random start page instead of --start.")
    parser.add_argument("--strategy", choices=["bfs", "best", "bidi"], default="bfs", help="Search strategy.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum clicks (depth) to attempt (default: 6).")
    parser.add_argument("--max-visited", type=int, default=50000, help="Safety cap on pages to visit.")
    parser.add_argument("--max-branch", type=int, default=50, help="For best-first: max neighbors to enqueue per expanded page.")
    parser.add_argument("--user-agent", help="Custom User-Agent header.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between API requests.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed crawl progress.")
    parser.add_argument("--flowchart", help="If set, save a PNG flowchart of the crawl to this filepath (e.g. ./graph.png).")
    args = parser.parse_args()

    crawler = WikiCrawler(
        user_agent=args.user_agent,
        sleep_between_requests=args.sleep,
        verbose=args.verbose
    )

    if args.random_start:
        start_title = crawler.random_page_title()
        print(f"Picked random start page: {start_title}")
    else:
        if not args.start:
            parser.error("Either --start or --random-start must be provided.")
        start_title = args.start

    try:
        resolved_target = crawler.resolve_title(args.target)
        if not resolved_target:
            print(f"Target '{args.target}' not found exactly; searching for best match...")
            resolved_target = crawler.search_title(args.target)
            if not resolved_target:
                raise SystemExit(f"Could not find a target page matching '{args.target}'.")
            print(f"Using target page: {resolved_target}")
        else:
            print(f"Target resolved to: {resolved_target}")

        resolved_start = crawler.resolve_title(start_title)
        if not resolved_start:
            print(f"Start '{start_title}' not found exactly; searching for best match...")
            resolved_start = crawler.search_title(start_title)
            if not resolved_start:
                raise SystemExit(f"Could not find a start page matching '{start_title}'.")
            print(f"Using start page: {resolved_start}")
        else:
            print(f"Start resolved to: {resolved_start}")

        if args.strategy == "bfs":
            path = crawler.find_path_bfs(resolved_start, resolved_target, max_depth=args.max_depth, max_visited=args.max_visited)
        elif args.strategy == "best":
            path = crawler.find_path_best_first(resolved_start, resolved_target, max_depth=args.max_depth, max_visited=args.max_visited, max_branch=args.max_branch)
        else:  # bidi
            path = crawler.find_path_bidi(resolved_start, resolved_target, max_depth=args.max_depth, max_visited=args.max_visited)

        if path:
            print("\n=== PATH FOUND ===")
            for i, t in enumerate(path):
                print(f"{i:2d}. {t}")
            print(f"Total clicks: {len(path)-1}")

            # draw flowchart if requested
            if args.flowchart:
                try:
                    crawler.draw_flowchart(args.flowchart, highlight_path=path, max_nodes=800)
                    print(f"Flowchart saved to: {args.flowchart}")
                except Exception as e:
                    print("Failed to draw flowchart:", e)
        else:
            print("\nNo path found within depth", args.max_depth)
            if args.flowchart:
                try:
                    crawler.draw_flowchart(args.flowchart, highlight_path=None, max_nodes=800)
                    print(f"Partial flowchart (no path) saved to: {args.flowchart}")
                except Exception as e:
                    print("Failed to draw flowchart:", e)

    except ValueError as ve:
        print("Error:", ve)

if __name__ == "__main__":
    main()

import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import "katex/dist/katex.min.css";
import submission from "../content/submission.md?raw";

export const Route = createFileRoute("/")({
  component: ReadingPage,
  head: () => ({
    meta: [
      { title: "FoodCourt+ Recommendation System" },
      {
        name: "description",
        content:
          "FoodCourt+ Recommendation System",
      },
    ],
    links: [
      {
        rel: "preconnect",
        href: "https://fonts.googleapis.com",
      },
      {
        rel: "preconnect",
        href: "https://fonts.gstatic.com",
        crossOrigin: "",
      },
      {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300..900;1,8..60,300..900&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap",
      },
    ],
  }),
});

import GithubSlugger from "github-slugger";

interface TocItem {
  level: 1 | 2;
  text: string;
  id: string;
}

function buildToc(md: string): TocItem[] {
  const lines = md.split("\n");
  const items: TocItem[] = [];
  const slugger = new GithubSlugger();
  let inCode = false;
  for (const line of lines) {
    if (line.startsWith("```")) inCode = !inCode;
    if (inCode) continue;
    const m1 = /^# (.+)$/.exec(line);
    const m2 = /^## (.+)$/.exec(line);
    if (m1) items.push({ level: 1, text: m1[1].trim(), id: slugger.slug(m1[1].trim()) });
    else if (m2) items.push({ level: 2, text: m2[1].trim(), id: slugger.slug(m2[1].trim()) });
  }
  return items;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="copy-btn"
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

function ReadingPage() {
  const toc = useMemo(() => buildToc(submission), []);
  const [activeId, setActiveId] = useState<string>(toc[0]?.id ?? "");
  const [progress, setProgress] = useState(0);
  const [navOpen, setNavOpen] = useState(false);
  const articleRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const total = h.scrollHeight - h.clientHeight;
      setProgress(total > 0 ? (h.scrollTop / total) * 100 : 0);

      // active section detection
      const headings = articleRef.current?.querySelectorAll("h1, h2");
      if (!headings) return;
      let current = activeId;
      const offset = 120;
      headings.forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.top <= offset) current = el.id;
      });
      if (current !== activeId) setActiveId(current);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [activeId]);

  return (
    <div className="min-h-screen">
      <div className="progress-bar" style={{ width: `${progress}%` }} />

      {/* Mobile top bar */}
      <div className="lg:hidden sticky top-0 z-40 flex items-center justify-between px-5 py-3 border-b" style={{ background: "var(--paper)", borderColor: "var(--rule)" }}>
        <span style={{ fontFamily: "var(--font-sans)", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--ink-muted)" }}>
          FoodCourt+
        </span>
        <button
          onClick={() => setNavOpen((v) => !v)}
          aria-label="Toggle navigation"
          style={{ fontFamily: "var(--font-sans)", color: "var(--accent)", fontSize: "0.85rem" }}
        >
          {navOpen ? "Close" : "Contents"}
        </button>
      </div>

      <div className="mx-auto max-w-[1200px] flex gap-8 px-5 lg:px-10">
        {/* Sidebar */}
        <aside
          className={`${navOpen ? "block" : "hidden"} lg:block lg:sticky lg:top-0 lg:self-start lg:h-screen lg:w-64 shrink-0 lg:overflow-y-auto py-10`}
        >
          <div className="mb-6 hidden lg:block">
            <div style={{ fontFamily: "var(--font-serif)", fontSize: "1.1rem", fontWeight: 700, color: "var(--ink)" }}>
              FoodCourt+
            </div>
          </div>
          <nav>
            {toc.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                onClick={() => setNavOpen(false)}
                className={`sidebar-link level-${item.level} ${activeId === item.id ? "active" : ""}`}
              >
                {item.text}
              </a>
            ))}
          </nav>
        </aside>

        {/* Article */}
        <article
          ref={articleRef}
          className="prose-paper flex-1 min-w-0 py-12 lg:py-20"
          style={{ maxWidth: "720px", margin: "0 auto" }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeRaw, rehypeSlug, [rehypeKatex, { strict: false, throwOnError: false }]]}
            components={{
              h1: ({ children, id, ...rest }) => (
                <h1 id={id} {...rest}>
                  {children}
                  <a href={`#${id}`} className="heading-anchor" aria-label="link">§</a>
                </h1>
              ),
              h2: ({ children, id, ...rest }) => (
                <h2 id={id} {...rest}>
                  {children}
                  <a href={`#${id}`} className="heading-anchor" aria-label="link">§</a>
                </h2>
              ),
              h3: ({ children, id, ...rest }) => (
                <h3 id={id} {...rest}>
                  {children}
                  <a href={`#${id}`} className="heading-anchor" aria-label="link">§</a>
                </h3>
              ),
              table: ({ children }) => (
                <div className="table-scroll">
                  <table>{children}</table>
                </div>
              ),
              pre: ({ children }) => {
                // children is a <code> element
                const child: any = Array.isArray(children) ? children[0] : children;
                const className: string = child?.props?.className || "";
                const lang = /language-(\w+)/.exec(className)?.[1];
                const codeText = extractText(child);
                return (
                  <div className="code-wrapper">
                    {lang && <span className="code-label">{lang}</span>}
                    <CopyButton text={codeText} />
                    <pre>{children}</pre>
                  </div>
                );
              },
            }}
          >
            {submission}
          </ReactMarkdown>

          <footer style={{ marginTop: "5rem", paddingTop: "2rem", borderTop: "1px solid var(--rule)", fontFamily: "var(--font-sans)", fontSize: "0.8rem", color: "var(--ink-muted)", textAlign: "center" }}>
            FoodCourt+
          </footer>
        </article>
      </div>
    </div>
  );
}

function extractText(node: any): string {
  if (node == null) return "";
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node.props?.children) return extractText(node.props.children);
  return "";
}

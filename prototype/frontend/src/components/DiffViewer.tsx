import React, { useMemo } from "react";

interface Props {
  preText?: string | null;
  postText?: string | null;
  title?: string;
}

interface DiffChunk {
  type: "same" | "del" | "ins";
  text: string;
}

export function computeWordDiff(oldStr: string, newStr: string): DiffChunk[] {
  if (!oldStr && !newStr) return [];
  if (!oldStr) return [{ type: "ins", text: newStr }];
  if (!newStr) return [{ type: "del", text: oldStr }];
  if (oldStr === newStr) return [{ type: "same", text: oldStr }];

  // Bound the LCS matrix; preserve the remainder as a coarse replacement.
  const maxTokens = 300;
  const tokenize = (text: string) => {
    const pattern = /\s+|[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu;
    const words: string[] = [];
    let end = 0;
    let match: RegExpExecArray | null;
    while (words.length < maxTokens && (match = pattern.exec(text))) {
      words.push(match[0]);
      end = pattern.lastIndex;
    }
    return { words, tail: text.slice(end) };
  };
  const oldTokens = tokenize(oldStr);
  const newTokens = tokenize(newStr);
  const oldWords = oldTokens.words;
  const newWords = newTokens.words;

  const dp: number[][] = Array(oldWords.length + 1)
    .fill(0)
    .map(() => Array(newWords.length + 1).fill(0));

  for (let i = 0; i < oldWords.length; i++) {
    for (let j = 0; j < newWords.length; j++) {
      if (oldWords[i] === newWords[j]) {
        dp[i + 1][j + 1] = dp[i][j] + 1;
      } else {
        dp[i + 1][j + 1] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  let i = oldWords.length;
  let j = newWords.length;
  const chunks: DiffChunk[] = [];

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldWords[i - 1] === newWords[j - 1]) {
      chunks.unshift({ type: "same", text: oldWords[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      chunks.unshift({ type: "ins", text: newWords[j - 1] });
      j--;
    } else if (i > 0) {
      chunks.unshift({ type: "del", text: oldWords[i - 1] });
      i--;
    }
  }

  if (oldTokens.tail === newTokens.tail) {
    if (oldTokens.tail) chunks.push({ type: "same", text: oldTokens.tail });
  } else {
    if (oldTokens.tail) chunks.push({ type: "del", text: oldTokens.tail });
    if (newTokens.tail) chunks.push({ type: "ins", text: newTokens.tail });
  }

  // Merge adjacent same-type chunks
  const merged: DiffChunk[] = [];
  for (const c of chunks) {
    if (merged.length > 0 && merged[merged.length - 1].type === c.type) {
      merged[merged.length - 1].text += c.text;
    } else {
      merged.push({ ...c });
    }
  }
  return merged;
}

export const DiffViewer: React.FC<Props> = ({
  preText,
  postText,
  title = "Output Comparison",
}) => {
  const chunks = useMemo(() => {
    return computeWordDiff(preText || "", postText || "");
  }, [preText, postText]);

  if (!preText && !postText) {
    return (
      <div className="diff-viewer empty">
        <div className="panel-header-sub">
          <span className="badge-tag">C</span>
          <h4>{title}</h4>
        </div>
        <p className="hint" style={{ padding: "0.8rem" }}>Run an edit to inspect token-level generation differences.</p>
      </div>
    );
  }

  return (
    <div className="diff-viewer">
      <div className="panel-header-sub">
        <span className="badge-tag">C</span>
        <h4>{title}</h4>
      </div>
      <div className="diff-body">
        {chunks.map((c, idx) => {
          if (c.type === "del") {
            return (
              <span key={idx} className="diff-del" title="Pre-edit text (removed)">
                {c.text}
              </span>
            );
          }
          if (c.type === "ins") {
            return (
              <span key={idx} className="diff-ins" title="Post-edit text (added)">
                {c.text}
              </span>
            );
          }
          return <span key={idx}>{c.text}</span>;
        })}
      </div>
    </div>
  );
};

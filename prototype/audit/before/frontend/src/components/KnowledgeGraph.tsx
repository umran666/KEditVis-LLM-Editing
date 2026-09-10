import React, { useMemo } from "react";

interface Props {
  subject: string;
  target?: string;
  onSelectSubject?: (subj: string) => void;
}

interface Node {
  id: string;
  label: string;
  x: number;
  y: number;
  type: "subject" | "target" | "neighbor";
}

interface Link {
  source: string;
  target: string;
  relation: string;
}

export const KnowledgeGraph: React.FC<Props> = ({ subject, target, onSelectSubject }) => {
  const { nodes, links } = useMemo(() => {
    const subjNode: Node = {
      id: "s0",
      label: subject || "Entity",
      x: 90,
      y: 65,
      type: "subject",
    };

    const targetNode: Node = {
      id: "t0",
      label: target || "Target",
      x: 150,
      y: 35,
      type: "target",
    };

    const neighbors: Node[] = [
      { id: "n1", label: "Location", x: 30, y: 40, type: "neighbor" },
      { id: "n2", label: "Category", x: 45, y: 95, type: "neighbor" },
      { id: "n3", label: "Neighbor", x: 145, y: 95, type: "neighbor" },
    ];

    const allNodes = [subjNode, targetNode, ...neighbors];

    const allLinks: Link[] = [
      { source: "s0", target: "t0", relation: "target" },
      { source: "s0", target: "n1", relation: "located_in" },
      { source: "s0", target: "n2", relation: "domain" },
      { source: "s0", target: "n3", relation: "related" },
    ];

    return { nodes: allNodes, links: allLinks };
  }, [subject, target]);

  return (
    <div className="knowledge-graph-box">
      <div className="panel-header-sub">
        <div className="title-with-badge">
          <span className="badge-tag">A2</span>
          <h4>Knowledge Graph</h4>
        </div>
      </div>
      <div className="kg-canvas-wrapper">
        <svg viewBox="0 0 180 130" className="kg-svg">
          {/* Links */}
          <g className="links">
            {links.map((l, i) => {
              const src = nodes.find((n) => n.id === l.source);
              const dst = nodes.find((n) => n.id === l.target);
              if (!src || !dst) return null;
              return (
                <g key={i}>
                  <line
                    x1={src.x}
                    y1={src.y}
                    x2={dst.x}
                    y2={dst.y}
                    stroke="#CBD5E1"
                    strokeWidth="1.2"
                    strokeDasharray={l.relation === "target" ? "3 2" : undefined}
                  />
                  <text
                    x={(src.x + dst.x) / 2}
                    y={(src.y + dst.y) / 2 - 3}
                    textAnchor="middle"
                    fill="#64748B"
                    fontSize="5"
                    fontFamily="var(--font-mono)"
                  >
                    {l.relation}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Nodes */}
          <g className="nodes">
            {nodes.map((n) => {
              const isSubj = n.type === "subject";
              const isTgt = n.type === "target";
              const fill = isSubj ? "#3B82F6" : isTgt ? "#10B981" : "#64748B";
              const r = isSubj ? 14 : isTgt ? 12 : 9;

              return (
                <g
                  key={n.id}
                  transform={`translate(${n.x}, ${n.y})`}
                  style={{ cursor: isSubj ? "default" : "pointer" }}
                  onClick={() => {
                    if (n.type === "neighbor" && onSelectSubject) {
                      onSelectSubject(n.label);
                    }
                  }}
                >
                  <title>{`Node: ${n.label} (${n.type})`}</title>
                  <circle
                    r={r}
                    fill={isSubj ? "#EFF6FF" : isTgt ? "#ECFDF5" : "#F1F5F9"}
                    stroke={fill}
                    strokeWidth="1.2"
                  />
                  <text
                    textAnchor="middle"
                    dy="2.5"
                    fill="#0F172A"
                    fontSize="6"
                    fontWeight={isSubj || isTgt ? 600 : 500}
                    fontFamily="var(--font-sans)"
                  >
                    {n.label.length > 9 ? n.label.slice(0, 8) + "…" : n.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
};

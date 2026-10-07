// Verified fixture screenshots are written to AUDIT_OUT, or audit/run by default.
console.log("Capturing dashboard fixture screenshots and verifying current UI behavior; no GPU editing is performed.");
await import("./tests/audit.mjs");

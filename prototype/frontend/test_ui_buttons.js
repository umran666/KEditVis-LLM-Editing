// Compatibility entrypoint: test the production components and button handlers.
console.log("Running current UI button regressions with API fixtures; no GPU editing is performed.");
await import("./tests/audit.mjs");

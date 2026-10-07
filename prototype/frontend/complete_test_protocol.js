// Compatibility entrypoint: exercise the current UI instead of copied mock logic.
console.log("Running the current browser fixture suite. Fixtures verify UI behavior; no GPU editing is performed.");
await import("./tests/audit.mjs");

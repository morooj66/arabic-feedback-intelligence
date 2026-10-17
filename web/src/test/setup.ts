import "@testing-library/jest-dom/vitest";

Object.defineProperty(window, "confirm", {
  writable: true,
  value: () => true,
});

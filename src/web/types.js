// Knowledge type colours: the single source of truth for every type colour in the web app.
// Graph nodes, the type filter, legends, badges and detail panels all read from here; no other
// file defines a type colour. Unknown or new types get the neutral fallback, never a generated one.
"use strict";

(function (global) {
  const TYPE_COLORS = Object.freeze({
    "document": "#64748B",            // slate
    "engineering concept": "#3B82F6", // blue
    "method": "#8B5CF6",              // purple
    "organization": "#0D9488",        // teal
    "process": "#D97706",             // amber
    "system component": "#16A34A",    // green
  });
  const ALL_TYPES_COLOR = "#6B7280";  // neutral grey: "All types" and every unknown type
  const FALLBACK_COLOR = ALL_TYPES_COLOR;
  const TINT_ALPHA = 0.14;            // badge and option backgrounds: a subtle tint of the type colour
  const MIN_CONTRAST = 4.5;           // WCAG AA for normal text

  const normalize = (type) => String(type ?? "").trim().toLowerCase();

  function hexToRgb(hex) {
    const value = parseInt(hex.slice(1), 16);
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
  }
  const rgbToHex = (rgb) => "#" + rgb.map((c) => Math.round(c).toString(16).padStart(2, "0")).join("").toUpperCase();
  const mix = (a, b, t) => a.map((c, i) => c + (b[i] - c) * t);
  function luminance(rgb) {
    const [r, g, b] = rgb.map((c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  function contrast(a, b) {
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high + 0.05) / (low + 0.05);
  }

  // Every shade is derived from the base colour, so one entry in TYPE_COLORS defines a whole type.
  function derive(base) {
    const rgb = hexToRgb(base);
    const white = [255, 255, 255], black = [0, 0, 0];
    const tint = mix(white, rgb, TINT_ALPHA);                  // the tint as it looks on a white surface
    let shade = 0, text = rgb;
    while (contrast(text, tint) < MIN_CONTRAST && shade < 1) { // darken the same hue until it is readable
      shade += 0.05;
      text = mix(rgb, black, shade);
    }
    return Object.freeze({
      color: base,
      background: rgbToHex(tint),
      text: rgbToHex(text),
      border: rgbToHex(mix(rgb, black, 0.25)),                  // node outline: separates nodes on the light canvas
      contrast: Math.round(contrast(text, tint) * 100) / 100,
    });
  }

  const STYLES = Object.freeze(Object.fromEntries(Object.entries(TYPE_COLORS).map(([type, hex]) => [type, derive(hex)])));
  const FALLBACK = derive(FALLBACK_COLOR);

  /** Colours of a type: {color, background, text, border, contrast}. "" or null means "all types". */
  function style(type) {
    return STYLES[normalize(type)] || FALLBACK;
  }

  const isKnown = (type) => Object.prototype.hasOwnProperty.call(STYLES, normalize(type));

  /** Inline CSS variables for an element showing a type; the stylesheet only refers to these variables. */
  function cssVars(type) {
    const s = style(type);
    return `--type-color:${s.color};--type-bg:${s.background};--type-fg:${s.text};--type-border:${s.border}`;
  }

  function setVars(element, type) {
    const s = style(type);
    element.style.setProperty("--type-color", s.color);
    element.style.setProperty("--type-bg", s.background);
    element.style.setProperty("--type-fg", s.text);
    element.style.setProperty("--type-border", s.border);
  }

  const escape = (text) => String(text ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

  /** A tinted badge with the type name. */
  function badge(type, label) {
    const name = label ?? (type || "concept");
    return `<span class="chip type-badge" data-type="${escape(normalize(type))}" style="${cssVars(type)}">${escape(name)}</span>`;
  }

  /** A small solid dot in the type colour, for compact places (lists, related concepts). */
  function dot(type) {
    return `<span class="type-dot" style="${cssVars(type)}" title="${escape(type || "unknown type")}"></span>`;
  }

  /** Colour each option of a type <select> and the select itself after the current choice. */
  function paintSelect(select) {
    for (const option of select.options) {
      const s = style(option.value);
      option.style.backgroundColor = s.background;
      option.style.color = s.text;
    }
    select.classList.add("type-select");
    setVars(select, select.value);
  }

  /** Legend for the given types (in that order), plus the neutral swatch for anything unknown. */
  function legend(types) {
    const items = types.map((type) =>
      `<li><span class="type-swatch" style="${cssVars(type)}"></span>${escape(type)}</li>`);
    return `<ul class="type-legend" aria-label="Concept type colours">${items.join("")}</ul>`;
  }

  global.KnowledgeTypes = Object.freeze({
    TYPE_COLORS, ALL_TYPES_COLOR, FALLBACK_COLOR, style, isKnown, cssVars, setVars, badge, dot, paintSelect, legend,
    contrast: (a, b) => contrast(hexToRgb(a), hexToRgb(b)),
  });
})(typeof window !== "undefined" ? window : globalThis);

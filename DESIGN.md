---
name: ReVoice-Lip｜唇声再生
description: A precise visual-speech recognition and prosody reconstruction workspace.
colors:
  primary: "oklch(0.55 0.16 68)"
  primary-hover: "oklch(0.49 0.15 68)"
  accent: "oklch(0.36 0.09 235)"
  background: "oklch(1 0 0)"
  surface: "oklch(0.975 0.004 250)"
  surface-strong: "oklch(0.94 0.008 250)"
  ink: "oklch(0.22 0.025 250)"
  muted: "oklch(0.41 0.025 250)"
  border: "oklch(0.88 0.012 250)"
  dark-background: "oklch(0.145 0.012 250)"
  dark-surface: "oklch(0.19 0.015 250)"
  dark-field: "oklch(0.225 0.016 250)"
  dark-ink: "oklch(0.93 0.008 250)"
  success: "oklch(0.49 0.12 155)"
  danger: "oklch(0.52 0.18 25)"
typography:
  headline:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.18rem"
    fontWeight: 720
    lineHeight: 1.15
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 650
    lineHeight: 1.35
  body:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 600
    lineHeight: 1.3
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  xxl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.background}"
    rounded: "{rounded.md}"
    padding: "12px 18px"
    height: "44px"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "{colors.background}"
    rounded: "{rounded.md}"
  field:
    backgroundColor: "{colors.background}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "10px 12px"
    height: "44px"
  workspace-panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "24px"
---

# Design System: ReVoice-Lip｜唇声再生

## Overview

**Creative North Star: "The Calibrated Workbench"**

The interface is a research instrument on a focused review desk: controlled,
legible, and ready for a live demonstration in either light or dark mode. A
restrained amber action color points to the next operation while a compact
toolbar keeps identity, model state, and the optional credential together. The
result column carries greater visual weight than configuration.

The system rejects decorative AI gradients, glass effects, nested cards, and
equal-weight control walls. Related controls stay close; distinct phases receive
generous separation. Motion communicates state only.

**Key Characteristics:**

- Full-width input/result split on desktop and a deliberate single-column flow on mobile.
- One warm action color, used sparingly for primary actions and selection.
- Compact adaptive toolbar and progressive disclosure for advanced controls.
- Quiet surfaces, explicit focus states, and evidence-first result presentation.
- Native light/dark surfaces with no cross-theme hard-coded panels.

## Colors

Honey amber supplies action energy while cool slate neutrals ground the
technical tool. Each semantic role has paired light and dark values.

### Primary

- **Signal Amber:** primary actions, selected navigation, and active controls.

### Secondary

- **Instrument Ink:** header structure, high-emphasis labels, and dark badges.

### Neutral

- **Bench White / Night Slate:** paired page backgrounds.
- **Cool Instrument Surface:** input rail and quiet grouped regions.
- **Graphite Ink:** primary copy with high contrast.
- **Slate Annotation:** secondary copy and help text only.

**The One Signal Rule.** Signal Amber occupies less than ten percent of a
screen and always means action or current selection.

## Typography

**Display Font:** Inter (with ui-sans-serif and system-ui fallbacks)  
**Body Font:** Inter (with ui-sans-serif and system-ui fallbacks)

**Character:** A single compact sans-serif family keeps English technical text,
paths, labels, and results coherent. Weight and spacing—not a second decorative
typeface—create hierarchy.

### Hierarchy

- **Headline** (720, 1.75rem, 1.15): one product title only.
- **Title** (650, 1rem, 1.35): workflow regions and result groups.
- **Body** (400, 0.9375rem, 1.55): instructions and explanatory copy, capped at 72ch.
- **Label** (600, 0.8125rem, 1.3): controls, diagnostics, and metadata.

**The Result Voice Rule.** Recognized text may be larger and heavier than body
copy; settings labels may not compete with it.

## Elevation

Depth is primarily tonal. Workspace rails and fields separate through surface
color and crisp borders; focused interaction uses a visible halo. The compact
toolbar is bordered rather than elevated.

### Shadow Vocabulary

- **Focus halo** (`0 0 0 3px oklch(0.55 0.16 68 / 0.20)`): keyboard focus only.

**The Flat Workbench Rule.** Content surfaces remain flat at rest. If every
region appears to float, the hierarchy has failed.

## Components

### Buttons

- **Shape:** controlled rounded rectangle (10px), minimum 44px tall.
- **Primary:** Signal Amber with white text and 12px × 18px padding.
- **Hover / Focus:** darker amber on hover; a visible amber halo on focus.
- **Secondary:** theme surface with an Instrument Ink label and quiet border.

### Cards / Containers

- **Corner Style:** gently curved (14px maximum).
- **Background:** quiet theme surface for input; primary theme surface for result output.
- **Shadow Strategy:** none at rest.
- **Border:** one full quiet border when required; never a colored side stripe.
- **Internal Padding:** 20–24px desktop, 16px mobile.

### Inputs / Fields

- **Style:** theme field surface, high-contrast text, quiet full border, 10px corners.
- **Focus:** amber border plus focus halo; placeholder text maintains contrast.
- **Error / Disabled:** error text includes a message; disabled state retains legibility.

### Navigation

Tabs use quiet text at rest and Signal Amber for the selected state. Each target
is at least 44px tall; active state never relies on color alone.

### Credential Field

The MiniMax connection group sits at the right side of the compact toolbar. It
combines a password field, account-region selector, non-billable verification
action, and inline connection result. Help text states that the value is
session-only and never persisted. The result rail separately names the provider
that actually generated each speech preview.

## Do's and Don'ts

### Do:

- **Do** follow input → recognize → inspect → listen in both desktop and mobile layouts.
- **Do** keep recognized text and audio more prominent than decoding parameters.
- **Do** use the 4/8/12/16/24/32/48px spacing rhythm consistently.
- **Do** preserve a visible focus state and 44px minimum action height.
- **Do** keep the mouth crop beside the transcript so evidence and result scan together.

### Don't:

- **Don't** recreate an unbalanced two-column form with empty space on one side.
- **Don't** use nested cards, decorative AI gradients, or glassmorphism.
- **Don't** mix light cards with dark controls or hard-code one theme into the other.
- **Don't** expose a full checkpoint path as primary status copy.
- **Don't** persist, echo, log, or prefill a user's MiniMax API key.
- **Don't** use a border-left or border-right greater than 1px as a colored accent.

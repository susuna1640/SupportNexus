# Warm Workspace UI

The frontend keeps its existing React component tree, page hierarchy, controls, and API behavior. This visual pass is deliberately limited to surface, color, type emphasis, and interaction-state styling.

## Reference-derived palette

- Canvas: `#f5f0e8` and `#fbf8f2` create the warm paper base.
- Surface: `#fffdf9` gives primary cards a clear but quiet lift.
- Accent: `#e56b48` is the coral action color for the product mark, primary actions, focus rings, and selected navigation.
- Supporting status: `#667755` is the muted botanical green used for healthy and successful states.
- Ink: `#2f2c29` and warm-gray supporting text replace the previous blue-gray console palette.

## Visual rules

- Keep density and page geometry intact so operational workflows do not change.
- Use thin warm-gray dividers and soft, low-elevation shadows instead of dark surface contrast.
- Give customer messages a quiet coral-tinted surface; assistant messages remain paper-white.
- Keep error states distinct but restrained, avoiding saturated blocks outside of primary actions.
- Preserve keyboard focus, disabled states, and the reduced-motion rule already present in the application.

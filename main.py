# Python g-code library
from pygcode import *

def serpentine(width, height, radius, stride, offset):
    """
    Create a serpentine along the x-axis
    :param width: Width of draw box
    :param height: Height of draw box
    :param radius: Border radius
    :param stride: Distance between rows
    :param offset: X-Y offset used as the origin
    :return: G-Code path
    """
    x, y = offset
    # Adjust x offset and width
    x += radius
    width -= radius

    codes = [
        GCodeRapidMove(X=x, Y=y + radius),
        GCodeLinearMove(Z=0)
    ]
    returning = False  # Return the print head to the starting side

    # Continue until there is no longer enough room to print
    while width > radius:
        codes += [
            GCodeLinearMove(X=x, Y=y + radius),
            GCodeLinearMove(Y=y + height - radius),
            GCodeLinearMove(X=x + stride),
            # Handle return path
            GCodeLinearMove(Y=y + radius)] if not returning else [
            GCodeLinearMove(X=x + stride),
            GCodeLinearMove(Y=y + height - radius),
        ]
        # Shrink available width and adjust x position
        width -= stride
        x += stride
        # Swap returning flag
        returning = not returning
    # Return the g-code
    return codes + [GCodeLinearMove(Z=5)]


# Preamble for the g-code script
preamble = [
    GCodeUseMillimeters(),
    GCodeAbsoluteDistanceMode(),
]

# Assemble two serpentines together
gcodes = (preamble + serpentine(36,14,2,2,(0,0))
          + serpentine(35,14,2,2,(1,0)))

# Print G-Codes
print('\n'.join(f'N{10 + i * 5} {str(g)}' for (i, g) in enumerate(gcodes)))

# PolyKin: A polymerization kinetics library for Python.
#
# Copyright Hugo Vale 2026

import base64
import io

import matplotlib.pyplot as plt


def to_html(plt_obj, fmt="png", dpi=100, transparent=True, **kwargs):
    r"""
    Generate an HTML img tag containing a Base64-encoded Matplotlib figure.

    This utility allows `markdown-exec` to capture the printed HTML string
    and render the image directly within the generated documentation.

    Parameters
    ----------
    plt_obj : matplotlib.pyplot or matplotlib.figure.Figure
        The Matplotlib object to render. Can be the global `plt` module
        or a specific figure instance.
    fmt : str, optional
        The image format to use (e.g., 'png', 'svg', 'jpg').
        The default is 'png'.
    dpi : int, optional
        The resolution in dots per inch. The default is 100.
    transparent : bool, optional
        If `True`, the axes patches will be transparent. Useful for
        documentation with dark/light mode toggles. The default is `True`.
    **kwargs : dict
        Additional keyword arguments passed to `savefig`.

    Returns
    -------
    str
        A string containing the HTML <img> tag with Base64 data.
    """
    buf = io.BytesIO()
    fig = plt_obj.gcf() if hasattr(plt_obj, "gcf") else plt_obj

    fig.savefig(
        buf, format=fmt, dpi=dpi, transparent=transparent, bbox_inches="tight", **kwargs
    )
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("ascii")

    return (
        f'<img src="data:image/{fmt};base64,{encoded}" '
        f'style="max-width:100%; height:auto; display:block; margin: 1em 0;" />'
    )

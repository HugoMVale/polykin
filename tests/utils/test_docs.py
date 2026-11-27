import matplotlib.pyplot as plt

from polykin.utils.docs import to_html


def test_to_html_svg_mime_type():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    html = to_html(fig, fmt="svg")

    assert html.startswith('<img src="data:image/svg+xml;base64,')
    assert "height:auto" in html


def test_to_html_png_mime_type():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    html = to_html(fig, fmt="png")

    assert html.startswith('<img src="data:image/png;base64,')


def test_to_html_optionally_keeps_figure_open():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    to_html(fig, close=False)

    assert plt.fignum_exists(fig.number)
    plt.close(fig)

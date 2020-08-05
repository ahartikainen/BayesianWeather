import matplotlib.pyplot as plt
import numpy as np


def create_geometry():
    fig, ax = plt.subplots(figsize=(5, 5))

    # base circle
    ax.add_artist(plt.Circle((-11, 0), radius=14, color="blue", alpha=1))

    # wedge
    dx = (14 ** 2 - 3 ** 2) ** 0.5
    dy = 3
    dx_2 = dx + 4.0
    dy_2 = 1.5
    ax.add_artist(
        plt.Polygon(
            [(dx - 11, dy), (dx_2 - 11, dy_2), (dx_2 - 11, -dy_2), (dx - 11, -dy),],
            alpha=1,
            color="skyblue",
        )
    )

    ax.add_artist(
        plt.Polygon(
            [(0 - 11, 0), (dx - 11, dy), (dx - 11, -dy),], alpha=1, color="royalblue"
        )
    )

    ax.plot([dx - 11, dx - 11], [dy, -dy], color="y", lw=1, marker="o", label="inner")
    ax.plot(
        [dx_2 - 11, dx_2 - 11],
        [dy_2, -dy_2],
        color="red",
        lw=1,
        marker="o",
        label="outer",
    )
    ax.plot(
        [dx - 11, dx_2 - 11],
        [0, 0],
        color="cyan",
        lw=1,
        marker="o",
        markeredgecolor="k",
        label="length",
    )

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis("off")
    fig.legend()


def calculate_bucket_area(radius, wedge_inner, wedge_outer, wedge_length):
    """Calculate bucket area
    
    Area circle + Area wedge
    """
    circumference = 2 * np.pi * radius
    wedge_angle = 2 * np.pi * wedge_inner / circumference
    wedge_area_outside = (
        wedge_outer * wedge_length + abs(wedge_inner - wedge_outer) * wedge_length
    )
    wedge_triangle_p = (radius * 2 + wedge_inner) / 2
    wedge_area_inside = np.sqrt(
        wedge_triangle_p
        * (wedge_triangle_p - radius) ** 2
        * (wedge_triangle_p - wedge_inner)
    )
    bucket_area_inside = (
        np.pi * radius ** 2
        - 0.5 * radius ** 2 * (wedge_angle - np.sin(wedge_angle))
        + np.sqrt(
            wedge_triangle_p
            * (wedge_triangle_p - radius) ** 2
            * (wedge_triangle_p - wedge_inner)
        )
    )
    bucket_area = (
        bucket_area_inside + wedge_area_inside + wedge_area_outside
    ) / 10_000  # m^2
    return bucket_area


def get_stan_data(**kwargs):
    """Transform dataframe data to (columns == groups) to Stan ragged data structure.
    
    Better format for data is where groups are one column, but this works for this example.
    """
    w_b = []
    s_b = []
    for g, ser in kwargs["weight_bucket"].iteritems():
        ser = ser.dropna()
        s_b.append(len(ser))
        w_b.extend(ser.values)
    N_b = len(w_b)

    w_bw = []
    s_bw = []
    for g, ser in kwargs["weight_bucket_water"].iteritems():
        ser = ser.dropna()
        s_bw.append(len(ser))
        w_bw.extend(ser.values)
    N_bw = len(w_bw)

    d = []
    s_d = []
    for g, ser in kwargs["diameter"].iteritems():
        ser = ser.dropna()
        s_d.append(len(ser))
        d.extend(ser.values)
    N_d = len(d)

    d_i = []
    s_di = []
    for g, ser in kwargs["wedge_inner"].iteritems():
        ser = ser.dropna()
        s_di.append(len(ser))
        d_i.extend(ser.values)
    N_di = len(d_i)

    d_o = []
    s_do = []
    for g, ser in kwargs["wedge_outer"].iteritems():
        ser = ser.dropna()
        s_do.append(len(ser))
        d_o.extend(ser.values)
    N_do = len(d_o)

    d_l = []
    s_dl = []
    for g, ser in kwargs["wedge_length"].iteritems():
        ser = ser.dropna()
        s_dl.append(len(ser))
        d_l.extend(ser.values)
    N_dl = len(d_l)

    prec_data = {
        "K": 2,
        "N_b": N_b,
        "w_b": w_b,
        "s_b": s_b,
        "N_bw": N_bw,
        "w_bw": w_bw,
        "s_bw": s_bw,
        "N_d": N_d,
        "d": d,
        "s_d": s_d,
        "N_di": N_di,
        "d_i": d_i,
        "s_di": s_di,
        "N_do": N_do,
        "d_o": d_o,
        "s_do": s_do,
        "N_dl": N_dl,
        "d_l": d_l,
        "s_dl": s_dl,
    }

    return prec_data

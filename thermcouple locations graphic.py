import matplotlib.pyplot as plt
from cycler import cycler

# Define variables
diameter = 400  # mm
oilLevel = 40
fins=int(input('fins?:'))
print(fins)
zwart = "black"
blauw = "#1f77b4"
oranje = "#ff7f0e"
groen = "#2ca02c"
rood = "#d62728"
paars = "#9467bd"
bruin = "#8c564b"
roos = "#e377c2"
grijs = "#7f7f7f"
khakigroen = "#bcbd22"
cyan = "#17becf"
colorlist = []
points = {
    "point0":  [195  , 9 ,  zwart ],
    "point1":  [155  , 9 ,  zwart ],
    "point2":  [115  , 9 ,  zwart ],
    "point3":  [75   , 9 ,  zwart ],
    "point4":  [35   , 9 ,  zwart ],
    "point5":  [12.5 , 9 ,  zwart ],
    "point6":  [35   , 24,  zwart ],
    "point7":  [75   , 24,  zwart ],
    "point8":  [115  , 24,  zwart ],
    "point9":  [155  , 24,  zwart ],
    "point10": [195  , 24,  zwart ],
    "point11": [195  , 24,  zwart ],
    "point12": [75   , 37,  zwart ],
    "point13": [115  , 37,  zwart ],
    "point14": [155  , 37,  zwart ],
    "point15": [195  , 37,  zwart ],
    "point17": [35   , 37,  zwart ],
}
# Create plot
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlabel("Radial Distance (mm)")
ax.set_ylabel("Axis Of Symmetry")
ax.set_xlim([0, diameter / 2 + 5])
ax.set_ylim([0, oilLevel * 4 / 3])
ax.set_yticks([])
for value in points.values():
    colorlist.append(value[2])
costum_cycler = cycler(color=colorlist)
ax.set_prop_cycle(costum_cycler)
for value in points.values():
    ax.plot([value[0]], [value[1]], "o")
ax.axhline(y=0, color="black", xmax=200)
ax.axvline(x=200, color="black")
if fins==1:

    color_fins="#7e7e7e"
    ax.axvline(x=16, color=color_fins)
    ax.axvline(x=85.5, color=color_fins)
    ax.axvline(x=125, color=color_fins)
    ax.axvline(x=159, color=color_fins)    #eig 157 ipv 160
    ax.axvline(x=191.2, color=color_fins)

ax.fill_betweenx([0, oilLevel], 0, 200, color="#e0e0e0")
ax.annotate("", xy=(0, oilLevel * 4 / 3 + 5), xytext=(0, -4), arrowprops=dict(arrowstyle="->", color="black",linestyle='--'), annotation_clip=False,)
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["left"].set_color("none")
plt.savefig("figuur2.pdf")
plt.show()



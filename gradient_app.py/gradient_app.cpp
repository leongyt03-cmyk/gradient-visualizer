import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go

st.set_page_config(page_title = "Gradient & Steepest Ascent Visualizer", layout = "wide")

st.title("🎯 Gradient & Steepest Ascent Visualizer")
st.markdown("Interactive visualization of gradient vectors and steepest ascent paths")

# Define functions and their gradients
functions = {
    "Paraboloid: f(x,y) = x² + y²": {
        "f": lambda x, y : x * *2 + y * *2,
        "grad_x" : lambda x, y : 2 * x,
        "grad_y" : lambda x, y : 2 * y,
    },
    "Saddle Point: f(x,y) = x² - y²" : {
        "f": lambda x, y : x * *2 - y * *2,
        "grad_x" : lambda x, y : 2 * x,
        "grad_y" : lambda x, y : -2 * y,
    },
    "Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²" : {
        "f": lambda x, y : (1 - x)** 2 + 100 * (y - x * *2) * *2,
        "grad_x" : lambda x, y : -2 * (1 - x) - 400 * x * (y - x * *2),
        "grad_y" : lambda x, y : 200 * (y - x * *2),
    },
    "Gaussian Hill: f(x,y) = e^(-(x²+y²))" : {
        "f": lambda x, y : np.exp(-(x * *2 + y * *2)),
        "grad_x" : lambda x, y : -2 * x * np.exp(-(x * *2 + y * *2)),
        "grad_y" : lambda x, y : -2 * y * np.exp(-(x * *2 + y * *2)),
    },
}

# Sidebar controls
st.sidebar.header("Controls")
selected_func = st.sidebar.selectbox("Select Function", list(functions.keys()))
show_gradients = st.sidebar.checkbox("Show Gradient Field", value = True)
show_contours = st.sidebar.checkbox("Show Contour Lines", value = True)
show_surface = st.sidebar.checkbox("Show 3D Surface", value = False)

st.sidebar.markdown("---")
st.sidebar.subheader("Starting Point")
start_x = st.sidebar.slider("Start X", -3.0, 3.0, -2.0, 0.1)
start_y = st.sidebar.slider("Start Y", -3.0, 3.0, -2.0, 0.1)
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 0.2, 0.05, 0.01)
max_steps = st.sidebar.slider("Max Steps", 50, 300, 150, 10)

# Get current function
func_dict = functions[selected_func]
f = func_dict["f"]
grad_x = func_dict["grad_x"]
grad_y = func_dict["grad_y"]

# Create grid
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Compute steepest ascent path
def compute_path(start_x, start_y, learning_rate, max_steps) :
    path_x = [start_x]
    path_y = [start_y]
    path_z = [f(start_x, start_y)]

    x_curr, y_curr = start_x, start_y

    for _ in range(max_steps) :
        gx = grad_x(x_curr, y_curr)
        gy = grad_y(x_curr, y_curr)
        magnitude = np.sqrt(gx * *2 + gy * *2)

        if magnitude < 0.001 :
            break

            x_curr += learning_rate * gx
            y_curr += learning_rate * gy

            if abs(x_curr) > 5 or abs(y_curr) > 5:
break

path_x.append(x_curr)
path_y.append(y_curr)
path_z.append(f(x_curr, y_curr))

return np.array(path_x), np.array(path_y), np.array(path_z)

path_x, path_y, path_z = compute_path(start_x, start_y, learning_rate, max_steps)

# Create visualizations
if show_surface:
# 3D Surface plot
fig = go.Figure(data = [go.Surface(x = X, y = Y, z = Z, colorscale = 'Viridis', opacity = 0.8)])

# Add path
fig.add_trace(go.Scatter3d(
    x = path_x, y = path_y, z = path_z,
    mode = 'lines+markers',
    line = dict(color = 'red', width = 6),
    marker = dict(size = 4, color = 'red'),
    name = 'Steepest Ascent Path'
))

# Add start point
fig.add_trace(go.Scatter3d(
    x = [start_x], y = [start_y], z = [f(start_x, start_y)],
    mode = 'markers',
    marker = dict(size = 10, color = 'lime', symbol = 'diamond'),
    name = 'Start Point'
))

fig.update_layout(
    title = selected_func,
    scene = dict(
        xaxis_title = 'X',
        yaxis_title = 'Y',
        zaxis_title = 'f(x,y)',
        camera = dict(eye = dict(x = 1.5, y = 1.5, z = 1.3))
    ),
    height = 600
)

st.plotly_chart(fig, use_container_width = True)

else:
# 2D Contour plot
fig, ax = plt.subplots(figsize = (10, 8))

if show_contours:
contour = ax.contour(X, Y, Z, levels = 20, cmap = 'Blues', alpha = 0.6)
ax.clabel(contour, inline = True, fontsize = 8)

if show_gradients :
    # Gradient field
    step = 8
    X_grad, Y_grad = np.meshgrid(x[::step], y[::step])
    U = grad_x(X_grad, Y_grad)
    V = grad_y(X_grad, Y_grad)
    ax.quiver(X_grad, Y_grad, U, V, color = 'red', alpha = 0.6, scale = 50, width = 0.003)

    # Plot path
    ax.plot(path_x, path_y, 'g-', linewidth = 3, label = 'Steepest Ascent Path')
    ax.plot(path_x, path_y, 'go', markersize = 4)

    # Start and end points
    ax.plot(start_x, start_y, 'ro', markersize = 12, label = 'Start Point')
    ax.plot(path_x[-1], path_y[-1], 'y*', markersize = 20, label = 'End Point')

    ax.set_xlabel('X', fontsize = 12)
    ax.set_ylabel('Y', fontsize = 12)
    ax.set_title(selected_func, fontsize = 14)
    ax.grid(True, alpha = 0.3)
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

    st.pyplot(fig)

    # Display information
    col1, col2, col3 = st.columns(3)

    with col1 :
st.metric("Starting Value", f"{f(start_x, start_y):.4f}")

with col2 :
st.metric("Final Value", f"{path_z[-1]:.4f}")

with col3 :
st.metric("Path Length", f"{len(path_x)} steps")

# Explanation section
st.markdown("---")
st.header("📚 How It Works")

col1, col2 = st.columns(2)

with col1 :
st.subheader("🔴 Gradient Vector")
st.markdown("""
    The gradient * *∇f = (∂f / ∂x, ∂f / ∂y) * *points in the direction of steepest ascent.
    - Red arrows show gradient direction at each point
    - Longer arrows = steeper slope
    - Always perpendicular to contour lines
    """)

    st.subheader("🟢 Steepest Ascent Algorithm")
    st.markdown("""
        Iteratively follows the gradient :

**x_{ n + 1 } = x_n + α∇f(x_n) * *

where α is the learning rate(step size)
""")

with col2 :
st.subheader("🔵 Contour Lines")
st.markdown("""
    Connect points with equal function values(like topographic maps).
    - Closer lines = steeper terrain
    - Gradient crosses contours at right angles
    """)

    st.subheader("🌍 Real-World Applications")
    st.markdown("""
        - **Machine Learning * *: Training neural networks(gradient descent)
        - **Robotics * *: Path planning on terrain
        - **Economics * *: Profit maximization
        - **Physics * *: Finding equilibrium states
        """)

        # Display current gradient
        st.markdown("---")
        st.subheader("Current Gradient at Start Point")
        gx_start = grad_x(start_x, start_y)
        gy_start = grad_y(start_x, start_y)
        magnitude = np.sqrt(gx_start * *2 + gy_start * *2)

        st.write(f"**∇f({start_x:.2f}, {start_y:.2f}) = ({gx_start:.4f}, {gy_start:.4f})**")
        st.write(f"**Magnitude: {magnitude:.4f}**")
        st.write(f"Direction points toward **{('higher' if magnitude > 0.01 else 'local maximum')} values**")
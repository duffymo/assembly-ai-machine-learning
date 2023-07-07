import matplotlib.pyplot as plt

from sklearn import datasets, manifold

def create_swiss_roll(n_samples, rs=1945, includeHole=False, noiseFraction=0.2):
    return datasets.make_swiss_roll(n_samples=n_samples, hole=includeHole, noise=noiseFraction, random_state=rs)

def plot_swiss_roll(n_samples, sr_points, sr_color):
    fig=plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.add_axes(ax)
    ax.scatter(
        sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8
    )
    ax.set_title('Swiss Roll In Ambient Space')
    ax.view_init(azim=66, elev=12)
    text = 'n_samples=' + str(n_samples)
    _ = ax.text2D(0.8, 0.05, s=text, transform=ax.transAxes)
    plt.show()

def plot_lle(sr_lle, sr_tsne, sr_color):
    fig, axs = plt.subplots(figsize=(8, 8), nrows=2)
    axs[0].scatter(sr_lle[:, 0], sr_lle[:, 1], c=sr_color)
    axs[1].scatter(sr_tsne[:, 0], sr_tsne[:, 1], c=sr_color)
    _ = axs[1].set_title('t-SNE Embedding of Swiss Roll')
    plt.show()

if __name__ == '__main__':
    n_samples = 1_000
    rs = 1945
    noiseFraction = 0.2
    n_components = 2

    sr_points, sr_color = create_swiss_roll(n_samples, rs, False, noiseFraction)
    plot_swiss_roll(n_samples, sr_points, sr_color)

    sr_lle, sr_err = manifold.locally_linear_embedding(sr_points, n_neighbors=12, n_components=n_components)
    sr_tsne = manifold.TSNE(n_components=n_components, perplexity=40, random_state=rs).fit_transform(sr_points)
    plot_lle(sr_lle, sr_tsne, sr_color)


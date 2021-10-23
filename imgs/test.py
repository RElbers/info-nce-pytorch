import matplotlib.pyplot as plt
import numpy as np
import torch

from info_nce import InfoNCE


# Numpy helper functions
def dot(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return x.dot(y.T)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy(x, eps=0.0):
    a = x[:, 0]
    b = np.sum(x, axis=1) + eps
    return -np.log((a / b) + eps)


def mu_sigma(dims):
    mu = np.random.rand(dims)

    # Random covariance matrix makes the plot too noisy
    # sigma = np.random.rand(dims, dims) * 0.1
    sigma = np.identity(dims) * 0.1
    return mu, sigma @ sigma.T


def gen_samples(mu, sigma, n):
    return np.random.multivariate_normal(mu, sigma, n)


def interpolate(x, y, weight):
    return weight * x + (1 - weight) * y


def infonce_manual(q_samples, p_samples, n_samples, temp):
    paired_negatives = n_samples.ndim == 3

    logits = []
    for i in range(len(q_samples)):
        q = q_samples[i]
        p = p_samples[i]
        sim_pos = dot(q, p)

        similarities = [sim_pos]
        if paired_negatives:
            for j in range(len(n_samples[i])):
                n = n_samples[i, j]
                sim_neg = dot(q, n)
                similarities.append(sim_neg)
        else:
            for j in range(len(n_samples)):
                n = n_samples[j]
                sim_neg = dot(q, n)
                similarities.append(sim_neg)

        logits.append(similarities)

    logits = np.array(logits) / temp
    loss = cross_entropy(softmax(logits))
    loss = np.mean(loss)
    return loss


def infonce_pytorch(q_samples, p_samples, n_samples, temp):
    q_samples = torch.from_numpy(q_samples)
    p_samples = torch.from_numpy(p_samples)
    n_samples = torch.from_numpy(n_samples)

    negative_mode = 'paired' if n_samples.ndim == 3 else 'unpaired'
    infonce = InfoNCE(temperature=temp, negative_mode=negative_mode)
    loss = infonce(q_samples, p_samples, n_samples)
    return loss.item()


def get_results(paired_neg, num_pos=16, num_neg=8, k=64, t=0.1):
    steps = 10
    manual_results, pytorch_results = [], []

    q_mu, q_sigma = mu_sigma(k)
    p_mu_init, p_sigma_init = mu_sigma(k)
    n_mu_init, n_sigma_init = mu_sigma(k)
    for alpha in np.linspace(0.0, 1.0, num=steps):
        for beta in np.linspace(1.0, 0.0, num=steps):
            # Move P closer to Q
            p_mu = interpolate(q_mu, p_mu_init, weight=alpha)
            p_sigma = interpolate(q_sigma, p_sigma_init, weight=alpha)
            # Move N further from Q
            n_mu = interpolate(q_mu, n_mu_init, weight=beta)
            n_sigma = interpolate(q_sigma, n_sigma_init, weight=beta)

            # Generate new samples
            q_samples = gen_samples(q_mu, q_sigma, num_pos)
            p_samples = gen_samples(p_mu, p_sigma, num_pos)
            if paired_neg:
                n_samples = [gen_samples(n_mu, n_sigma, num_neg) for _ in range(num_pos)]
                n_samples = np.array(n_samples)
            else:
                n_samples = gen_samples(n_mu, n_sigma, num_neg)

            # Test InfoNCE
            manual_loss = infonce_manual(q_samples, p_samples, n_samples, t)
            pytorch_loss = infonce_pytorch(q_samples, p_samples, n_samples, t)

            manual_results.append([alpha, beta, manual_loss])
            pytorch_results.append([alpha, beta, pytorch_loss])

    return manual_results, pytorch_results


def plot_results(results):
    data = np.array(results)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure()
    fig.set_size_inches(8,8)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=-70, elev=30)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel('Loss')
    ax.plot_trisurf(x, y, z, cmap='viridis')
    plt.savefig('./loss.png', dpi=100)
    plt.show()


def check_results(manual_results, pytorch_results):
    for manual, pytorch in zip(manual_results, pytorch_results):
        manual_loss, pytorch_loss = manual[2], pytorch[2]
        assert (abs(manual_loss - pytorch_loss) < 0.00001)

    print("All values are close.")
    plot_results(pytorch_results)


if __name__ == '__main__':
    paired_results = get_results(paired_neg=True)
    unpaired_results = get_results(paired_neg=False)

    check_results(*paired_results)
    check_results(*unpaired_results)

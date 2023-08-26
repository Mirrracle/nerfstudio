import torch

torch.manual_seed(27)

# X = torch.tensor([1.0, 2.0, 3.0, 4.0])
# Y = 2.0 * X + 1.0
# a = torch.tensor([5.0, 6.0, 7.0, 8.0])
# b = 2.0 * a + 1.0
# X = torch.cat((X, a), dim=0)
# Y = torch.cat((Y, b), dim=0)

# torch.save(X, 'X.pt')
# torch.save(Y, 'Y.pt')
X = torch.load('pred_3.pt').double()
Y = torch.load('gt_3.pt').double()

X = X.unsqueeze(0).T
Y = Y.unsqueeze(0).T
X = torch.cat((X, torch.ones(X.shape[0], 1)), dim=1)

A = torch.linalg.lstsq(X, Y).solution
# A = torch.inverse(X.T @ X) @ X.T @ X_pred
print(A)
# print(X @ A)

torch.save(A, 'depth_transformation_3.pt')




from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

n=input('show image i: Please enter a number: ')
n=int(n)
some_digit = X[n]
para1 = int(some_digit.shape[0]**0.5)
some_digit_image = some_digit.reshape(para1,para1)
plt.imshow(some_digit_image,cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.show()

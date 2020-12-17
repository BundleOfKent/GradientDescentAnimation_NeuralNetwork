# In[1]:
# Import libraries
import numpy as np
import gzip
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import expit
import celluloid
from celluloid import Camera
from matplotlib import animation 

# Open MNIST-files: 
def open_images(filename):
    with gzip.open(filename, "rb") as file:     
        data=file.read()
        return np.frombuffer(data,dtype=np.uint8, offset=16).reshape(-1,28,28).astype(np.float32) 

def open_labels(filename):
    with gzip.open(filename,"rb") as file:
        data = file.read()
        return np.frombuffer(data,dtype=np.uint8, offset=8).astype(np.float32) 
    
X_train=open_images("C:\\Users\\tobia\\train-images-idx3-ubyte.gz").reshape(-1,784).astype(np.float32) 
X_train=X_train/255 # rescale pixel values to 0-1

y_train=open_labels("C:\\Users\\tobia\\train-labels-idx1-ubyte.gz")
oh=OneHotEncoder(categories='auto') 
y_train_oh=oh.fit_transform(y_train.reshape(-1,1)).toarray() # one-hot-encoding of y-values




# In[2]:
hidden_0=50 # number of nodes of first hidden layer
hidden_1=500 # number of nodes of second hidden layer

# Set up cost function:
def costs(x,y,w_a,w_b,seed_):  
        np.random.seed(seed_) # insert random seed 
        w0=np.random.randn(hidden_0,784)  # weight matrix of 1st hidden layer
        w1=np.random.randn(hidden_1,hidden_0) # weight matrix of 2nd hidden layer
        w2=np.random.randn(10,hidden_1) # weight matrix of output layer
        w2[5][250] = w_a # set value for weight w_250,5(2)
        w2[5][251] = w_b # set value for weight w_251,5(2)
        a0 = expit(w0 @ x.T)  # output of input layer
        a1=expit(w1 @ a0)  # output of 1st hidden layer
        pred= expit(w2 @ a1) # output of 2nd hidden layer
        return np.mean(np.sum((y-pred)**2,axis=0)) # costs w.r.t. w_a and w_b




# In[3]:
# Set range of values for meshgrid: 
m1s = np.linspace(-15, 17, 40)   
m2s = np.linspace(-15, 18, 40)  
M1, M2 = np.meshgrid(m1s, m2s) # create meshgrid 

# Determine costs for each coordinate in meshgrid: 
zs_100 = np.array([costs(X_train[0:100],y_train_oh[0:100].T  
                               ,np.array([[mp1]]), np.array([[mp2]]),135)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_100 = zs_100.reshape(M1.shape) # z-values for N=100

zs_10000 = np.array([costs(X_train[0:10000],y_train_oh[0:10000].T  
                               ,np.array([[mp1]]), np.array([[mp2]]),135)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_10000 = zs_10000.reshape(M1.shape) # z-values for N=10,000



# In[4]:
fig = plt.figure(figsize=(10,7.5)) # create figure
ax0 = fig.add_subplot(121, projection='3d' )
ax1 = fig.add_subplot(122, projection='3d' )

fontsize_=20 # set axis label fontsize
labelsize_=12 # set tick label size

# Customize subplots: 
ax0.view_init(elev=30, azim=-20)
ax0.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=9)
ax0.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-5)
ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
ax0.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
ax0.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
ax0.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
ax0.set_title('N:100',y=0.85,fontsize=15) # set title of subplot 

ax1.view_init(elev=30, azim=-30)
ax1.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=9)
ax1.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-5)
ax1.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
ax1.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
ax1.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
ax1.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
ax1.set_title('N:10,000',y=0.85,fontsize=15)

# Surface plots of costs (= loss landscapes):  
ax0.plot_surface(M1, M2, Z_100, cmap='terrain', #surface plot
                             antialiased=True,cstride=1,rstride=1, alpha=0.75)
ax1.plot_surface(M1, M2, Z_10000, cmap='terrain', #surface plot
                             antialiased=True,cstride=1,rstride=1, alpha=0.75)
plt.tight_layout()
plt.show()



# In[5]:
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d' ) 
ax.view_init(elev=30, azim=-15)
fontsize_=25
labelsize_= 15
ax.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=11)
ax.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=0)
ax.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
ax.tick_params(axis='x', pad=7, which='major', labelsize=labelsize_)
ax.tick_params(axis='y', pad=-3, which='major', labelsize=labelsize_)
ax.tick_params(axis='z', pad=7, which='major', labelsize=labelsize_)
ax.plot_surface(M1, M2, Z_100, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.69)
plt.tight_layout()
plt.savefig('NN_loss.png') # save plot as ".png"
plt.show()




# In[6]:
# Store values of costs and weights in lists: 
weights_2_5_250=[] 
weights_2_5_251=[] 
costs=[] 

seed_= 135 # random seed
N=100 # sample size 

# Set up neural network: 
class NeuralNetwork(object):
    def __init__(self, lr=0.01):
        self.lr=lr
        np.random.seed(seed_) # set random seed
        # Intialize weight matrices: 
        self.w0=np.random.randn(hidden_0,784)  
        self.w1=np.random.randn(hidden_1,hidden_0)
        self.w2=np.random.randn(10,hidden_1)
        self.w2[5][250] = start_a # set starting value for w_a
        self.w2[5][251] = start_b # set starting value for w_b
    
    def train(self, X,y):
        a0 = expit(self.w0 @ X.T)  
        a1=expit(self.w1 @ a0)  
        pred= expit(self.w2 @ a1)
        # Partial derivatives of costs w.r.t. weights of the 2nd hidden layer: 
        dw2= (pred - y.T)*pred*(1-pred)  @ a1.T / len(X)   # ... averaged over the sample size
        # Update weights: 
        self.w2[5][250]=self.w2[5][250] - self.lr * dw2[5][250] 
        self.w2[5][251]=self.w2[5][251] - self.lr * dw2[5][251] 
        costs.append(self.cost(pred,y)) # append cost values to list
    
    def cost(self, pred, y):
        return np.mean(np.sum((y.T-pred)**2,axis=0))
    
# Initial values of w_a/w_b: 
starting_points = [  (-9,15),(-10.1,15),(-11,15)] 

for j in starting_points:
    start_a,start_b=j
    model=NeuralNetwork(10) # set learning rate to 10
    for i in range(10000):  # 10,000 epochs            
        model.train(X_train[0:N], y_train_oh[0:N]) 
        weights_2_5_250.append(model.w2[5][250]) # append weight values to list
        weights_2_5_251.append(model.w2[5][251]) # append weight values to list

# Create sublists of costs and weight values for each starting point: 
costs = np.split(np.array(costs),3) 
weights_2_5_250 = np.split(np.array(weights_2_5_250),3)
weights_2_5_251 = np.split(np.array(weights_2_5_251),3)




# ## Surface plot: 
# In[7]:
fig = plt.figure(figsize=(10,10)) # create figure
ax = fig.add_subplot(111,projection='3d' ) 
line_style=["dashed", "dashdot", "dotted"] #linestyles
fontsize_=27 # set axis label fontsize
labelsize_=17 # set tick label fontsize
ax.view_init(elev=30, azim=-10)
ax.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=17)
ax.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=5)
ax.set_zlabel("costs", fontsize=fontsize_, labelpad=-35)
ax.tick_params(axis='x', pad=12, which='major', labelsize=labelsize_)
ax.tick_params(axis='y', pad=0, which='major', labelsize=labelsize_)
ax.tick_params(axis='z', pad=8, which='major', labelsize=labelsize_)
ax.set_zlim(4.75,4.802) # set range for z-values in the plot

# Define which epochs to plot:
p1=list(np.arange(0,200,20))
p2=list(np.arange(200,9000,100))
points_=p1+p2

camera=Camera(fig) # create Camera object
for i in points_:
    # Plot the three trajectories of gradient descent...
    #... each starting from its respective starting point
    #... and each with a unique linestyle:
    for j in range(3): 
        ax.plot(weights_2_5_250[j][0:i],weights_2_5_251[j][0:i],costs[j][0:i],
                linestyle=line_style[j],linewidth=2,
                color="black", label=str(i))
        ax.scatter(weights_2_5_250[j][i],weights_2_5_251[j][i],costs[j][i],
                   marker='o', s=15**2,
               color="black", alpha=1.0)
    # Surface plot (= loss landscape):
    ax.plot_surface(M1, M2, Z_100, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.75)
    ax.legend([f'epochs: {i}'], loc=(0.25, 0.8),fontsize=17) # set position of legend
    plt.tight_layout() 
    camera.snap() # take snapshot after each iteration
    
animation = camera.animate(interval = 5, # set delay between frames in milliseconds
               #           repeat = False,
                    #      repeat_delay = 0)
animation.save('gd_1.gif', writer = 'imagemagick', dpi=100)  # save animation      




# ## Contour plot: 
# In[8]:
fig = plt.figure(figsize=(10,10)) # create figure
ax0=fig.add_subplot(2, 1, 1) 
ax1=fig.add_subplot(2, 1, 2) 

# Customize subplots: 
ax0.set_xlabel(r'$w_a$', fontsize=25, labelpad=0)
ax0.set_ylabel(r'$w_b$', fontsize=25, labelpad=-20)
ax0.tick_params(axis='both', which='major', labelsize=17)
ax1.set_xlabel("epochs", fontsize=22, labelpad=5)
ax1.set_ylabel("costs", fontsize=25, labelpad=7)
ax1.tick_params(axis='both', which='major', labelsize=17)

contours_=21 # set the number of contour lines
points_=np.arange(0,9000,100) # define which epochs to plot

camera = Camera(fig) # create Camera object
for i in points_:
    cf=ax0.contour(M1, M2, Z_100,contours_, colors='black', # contour plot
                     linestyles='dashed', linewidths=1)
    ax0.contourf(M1, M2, Z_100, alpha=0.85,cmap='terrain') # filled contour plots 
    
    for j in range(3):
        ax0.scatter(weights_2_5_250[j][i],weights_2_5_251[j][i],marker='o', s=13**2,
               color="black", alpha=1.0)
        ax0.plot(weights_2_5_250[j][0:i],weights_2_5_251[j][0:i],
                linestyle=line_style[j],linewidth=2,
                color="black", label=str(i))
        
        ax1.plot(costs[j][0:i], color="black", linestyle=line_style[j])
    plt.tight_layout()
    camera.snap()
    
animation = camera.animate(interval = 5,
                    #      repeat = True, repeat_delay = 0)  # create animation 
animation.save('gd_2.gif', writer = 'imagemagick')  # save animation as gif




# ## Create some loss landscapes w.r.t. random seed: 
# In[9]:
N=1000 
hidden_0=50
hidden_1=500
contours= 21

for j in range(100):  
    seed_=j
    print("Seed: " + str(j))
    
    def costs(x,y,w_a,w_b):
        
        np.random.seed(seed_) 
        w0=np.random.randn(hidden_0,784) 
        w1= np.random.randn(hidden_1,hidden_0)
        w2=np.random.randn(10,hidden_1) 
        w1[5][5] = w_a
        w1[5][6] = w_b
        a0 = expit(w0 @ x.T)  
        a1= expit(w1@a0)
        pred= expit(w2 @ a1) 
        return np.mean(np.sum((y.T-pred)**2,axis=0))
    
    m1s = np.linspace(-15, 17, 25)   
    m2s = np.linspace(-15, 18, 25)
    M1, M2 = np.meshgrid(m1s, m2s)
    zs = np.array([costs(X_train[0:N],y_train_oh[0:N]  
                               ,np.array([[mp1]]), np.array([[mp2]]))  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
    Z = zs.reshape(M1.shape) 

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d' ) 
    ax.view_init(elev=40, azim=220)
    ax.plot_surface(M1, M2, Z, cmap='terrain', #surface plot
                             antialiased=True,cstride=1,rstride=1, alpha=0.69)
    plt.tight_layout()
    plt.show() 





# ## Representative loss landscapes: 
# In[10]:

from matplotlib import animation 
N=1000 # new sample size (N=1000)

# Define new cost function w.r.t. new weights of second hidden layer:
def costs_2(x,y,w_a,w_b, seed_):
        np.random.seed(seed_) 
        w0=np.random.randn(hidden_0,784) 
        w1= np.random.randn(hidden_1,hidden_0)
        w2=np.random.randn(10,hidden_1) 
        w1[5][5] = w_a # w5–5(1)
        w1[5][6] = w_b # w5–6(1)
        a0 = expit(w0 @ x.T)  
        a1= expit(w1@a0)
        pred= expit(w2 @ a1) 
        return np.mean(np.sum((y.T-pred)**2,axis=0))

# Calculate z-values w.r.t. random seed with new cost function:
zs_158 = np.array([costs_2(X_train[0:N],y_train_oh[0:N] 
                               ,np.array([[mp1]]), np.array([[mp2]]),158)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_158 = zs_158.reshape(M1.shape)

zs_20 = np.array([costs_2(X_train[0:N],y_train_oh[0:N]  
                               ,np.array([[mp1]]), np.array([[mp2]]),20)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_20 = zs_20.reshape(M1.shape)

zs_41 = np.array([costs_2(X_train[0:N],y_train_oh[0:N]  
                               ,np.array([[mp1]]), np.array([[mp2]]),41)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_41 = zs_41.reshape(M1.shape)

zs_106 = np.array([costs_2(X_train[0:N],y_train_oh[0:N]   
                               ,np.array([[mp1]]), np.array([[mp2]]),140)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_106 = zs_106.reshape(M1.shape)

fontsize_=19 # axis label font size
titlefontsize_=16 # subplot title font size

# Add subplots to figure: 
fig = plt.figure(figsize=(8.2,8.2))
ax0 = fig.add_subplot(2, 2, 1,projection='3d' )
ax1=fig.add_subplot(2, 2, 2,projection='3d') 
ax2=fig.add_subplot(2, 2, 3,projection='3d') 
ax3=fig.add_subplot(2, 2, 4,projection='3d') 

# Customize subplots:
ax0.set_title('seed:158', fontsize=titlefontsize_)
ax0.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax0.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax0.set_xticklabels([]) # remove axis tick labels
ax0.set_yticklabels([])
ax0.set_zticklabels([])
ax1.set_title('seed:20', fontsize=titlefontsize_)
ax1.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax1.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax1.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax2.set_title('seed:41', fontsize=titlefontsize_)
ax2.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax2.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax2.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax3.set_title('seed:106', fontsize=titlefontsize_)
ax3.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax3.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax3.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_zticklabels([])

# Rotate plots around the z-axis: 
def rotate(angle):
    ax0.view_init(elev=50,azim=angle)
    ax1.view_init(elev=50,azim=angle)
    ax2.view_init(elev=50,azim=angle)
    ax3.view_init(elev=50,azim=angle)

# Create loss landscapes w.r.t. seed: 
ax0.plot_surface(M1, M2, Z_158, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
ax1.plot_surface(M1, M2, Z_20, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
ax2.plot_surface(M1, M2, Z_41, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
ax3.plot_surface(M1, M2, Z_106, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
plt.tight_layout()

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)  
rot_animation.save('RotLoss_1000.gif', dpi=80, writer='imagemagick') 





# In[11]:
N=500 # new sample size (N=500)

# Define new cost function w.r.t. new weights of second hidden layer:
def costs_3(x,y,w_a,w_b,seed_):   
        np.random.seed(seed_) 
        w0=np.random.randn(hidden_0,784) 
        w1= np.random.randn(hidden_1,hidden_0)
        w2=np.random.randn(10,hidden_1) 
        w1[200][30] = w_a # w200–30(1)
        w1[200][31] = w_b # w200–31(1)
        a0 = expit(w0 @ x.T)  
        a1= expit(w1@a0)
        pred= expit(w2 @ a1) 
        return np.mean(np.sum((y.T-pred)**2,axis=0))

# Calculate z-values w.r.t. random seed with new cost function:
zs_3 = np.array([costs_3(X_train[0:N],y_train_oh[0:N]  
                               ,np.array([[mp1]]), np.array([[mp2]]),3)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_3 = zs_3.reshape(M1.shape)

zs_16 = np.array([costs_3(X_train[0:N],y_train_oh[0:N]   
                               ,np.array([[mp1]]), np.array([[mp2]]),16)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_16 = zs_16.reshape(M1.shape)

zs_60 = np.array([costs_3(X_train[0:N],y_train_oh[0:N]  
                               ,np.array([[mp1]]), np.array([[mp2]]),60)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_60 = zs_60.reshape(M1.shape)

zs_140 = np.array([costs_3(X_train[0:N],y_train_oh[0:N]  
                               ,np.array([[mp1]]), np.array([[mp2]]),140)  
                       for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
Z_140 = zs_140.reshape(M1.shape)

titlefontsize_=16
fontsize_=19

# Add subplots to figure: 
fig = plt.figure(figsize=(8,8)) 
ax0 = fig.add_subplot(2, 2, 1,projection='3d' )
ax1=fig.add_subplot(2, 2, 2,projection='3d') 
ax2=fig.add_subplot(2, 2, 3,projection='3d') 
ax3=fig.add_subplot(2, 2, 4,projection='3d') 

# Customize subplots:
ax0.set_title('Seed:3', fontsize=titlefontsize_)
ax0.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax0.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.set_zticklabels([])
ax1.set_title('Seed:16', fontsize=titlefontsize_)
ax1.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax1.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax1.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])
ax2.set_title('Seed:60', fontsize=titlefontsize_)
ax2.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax2.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax2.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])
ax3.set_title('Seed:140', fontsize=titlefontsize_)
ax3.set_xlabel(r'$w_a$', fontsize=fontsize_, labelpad=-4)
ax3.set_ylabel(r'$w_b$', fontsize=fontsize_, labelpad=-9)
ax3.set_zlabel("costs", fontsize=fontsize_, labelpad=-7)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_zticklabels([])

# Rotate plots around the z-axis: 
def rotate(angle):
    ax0.view_init(elev=50,azim=angle)
    ax1.view_init(elev=50,azim=angle)
    ax2.view_init(elev=50,azim=angle)
    ax3.view_init(elev=50,azim=angle)

# Create loss landscapes w.r.t. seed: 
ax0.plot_surface(M1, M2, Z_3, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
ax1.plot_surface(M1, M2, Z_16, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
ax2.plot_surface(M1, M2, Z_60, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
ax3.plot_surface(M1, M2, Z_140, cmap='terrain', 
                             antialiased=True,cstride=1,rstride=1, alpha=0.99)
plt.tight_layout()

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
rot_animation.save('RotLoss_500.gif', dpi=80, writer='imagemagick')


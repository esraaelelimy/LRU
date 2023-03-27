import sys
sys.path.append('./')
sys.path.append('../')

import jax.numpy as jnp
import jax
from typing import Callable, Any, Tuple
import flax.linen as nn


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  
Array = Any


class LRUCell(nn.Module):
    param_dtype: Dtype = jnp.float32
    r_max: jnp.float32 = 1.0
    r_min: jnp.float32 = 0.0    
    max_phase : jnp.float32 = 6.28
    @nn.compact
    def __call__(self, x_kminus1,u_k):
        input_dim = u_k.shape[-1]
        hidden_dim = x_kminus1.shape[-1]
        out_dim = input_dim
        nu_log = self.param('nu_log', self.nu_log_init, (hidden_dim,),self.r_max,self.r_min)
        theta_log = self.param('theta_log', self.theta_log_init, (hidden_dim,),self.max_phase)
        gamma_log = self.param('gamma_log', self.gamma_log_init, nu_log,theta_log)
        
        B_real = self.param('B_real',self.B_real_init,(input_dim,hidden_dim))
        
        B_img =  self.param('B_img',self.B_img_init,(input_dim,hidden_dim))
        
        C_real = self.param('C_real',
                    self.C_real_init,(hidden_dim,out_dim))

        C_img =  self.param('C_img',
                            self.C_img_init,(hidden_dim,out_dim))
        
        D = self.param('D',self.D_init,(input_dim,))        
        
        
        Lambda = jnp.exp(-jnp.exp(nu_log) + 1j* jnp.exp(theta_log))
        B = (B_real + 1j * B_img)
        B_norm = B * jnp.exp(jnp.expand_dims(gamma_log,axis=0))
        C = (C_real + 1j * C_img)
        
        x_k = (Lambda * x_kminus1) +  (u_k @ B_norm) 
        y_k =  (x_k @ C).real  + (D * u_k)
         
        return x_k,y_k
        
    """
    nu_log_init() and theta_log_init() initializes the eigen values on a ring in the complex plane.
    """
    @staticmethod
    def nu_log_init(key,shape,r_max = 1 ,r_min = 0):
        u1 = jax.random.uniform(key, shape=shape)
        nu_log = jnp.log(-0.5*jnp.log(u1*(r_max**2 - r_min**2) + r_min**2))
        return nu_log

    @staticmethod
    def theta_log_init(key,shape, max_phase = 6.28):
        u2 = jax.random.uniform(key, shape=shape)
        theta_log = jnp.log(max_phase*u2)
        return theta_log

    @staticmethod    
    def gamma_log_init(key,nu_log,theta_log):
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j* jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda)**2)) 
        return gamma_log
    
    @staticmethod    
    def B_real_init(key,shape):
        return jax.random.normal(key,shape=shape) / jnp.sqrt(2*shape[-1])
    
    @staticmethod    
    def B_img_init(key,shape):
        return jax.random.normal(key,shape=shape) / jnp.sqrt(2*shape[-1])
    
    @staticmethod    
    def C_real_init(key,shape):
        return jax.random.normal(key,shape=shape) / jnp.sqrt(shape[-1])

    @staticmethod    
    def C_img_init(key,shape):
        return jax.random.normal(key,shape=shape) / jnp.sqrt(shape[-1])
    
    @staticmethod    
    def D_init(key,shape):
        return jax.random.normal(key,shape=shape) / jnp.sqrt(2*shape[-1])
      

import sys
sys.path.append('./')
sys.path.append('../')

import jax.numpy as jnp
import jax
from typing import Callable, Any, Tuple
from flax import linen as nn
import flax.linen as nn
from src.lru.LRUCell import LRUCell

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  
Array = Any


class LRU(nn.Module):
    param_dtype: Dtype = jnp.float32
    r_max: jnp.float32 = 1.0
    r_min: jnp.float32 = 0.0    
    max_phase : jnp.float32 = 6.28
    @nn.compact
    def __call__(self, x_kminus1,u_k):
        model = LRUCell(param_dtype=self.param_dtype,
                        r_max=self.r_max,
                        r_min=self.r_min,
                        max_phase=self.max_phase)
        
        def model_fn(model,x_kminus1,u_k):
            x_k, out_k = model(x_kminus1,u_k)
            return x_k, out_k
        
        lru_scan = nn.scan(
            model_fn, 
            variable_broadcast="params",
            split_rngs={"params": False}, 
            in_axes=1, 
            out_axes=1)

        x_k, outs = lru_scan(model,x_kminus1,u_k)
        return x_k,outs

def test_lru():
    mod = LRU()
    dummy_x0 = jnp.ones((1,5),dtype=jnp.complex64)
    dummy_u0 = jnp.ones((1,1,3))
    key = jax.random.PRNGKey(0)
    variables = mod.init(key,dummy_x0,dummy_u0)
    
    input_seq = jnp.ones(shape=(1,10,3))
    x_0 = jnp.zeros(shape=(1,5),dtype=jnp.complex64)
    
    x_k,y_k = mod.apply(variables,x_0,input_seq)
    print(y_k)
    print(x_k)
if __name__== "__main__":
    test_lru()
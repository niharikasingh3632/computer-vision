import numpy as np
import time

class Value:
    def __init__(self, data, _children=(), _op='', _label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._label = _label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, _children=(self, other), _op='-')  # Fixed: was +(-1 * other.data)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __pow__(self, other):
        if isinstance(other, Value):
            # Handle Value ** Value case
            out = Value(self.data ** other.data, _children=(self, other), _op='**')
            
            def _backward():
                # d/dx (x^y) = y * x^(y-1)
                self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
                # d/dy (x^y) = x^y * ln(x)
                other.grad += out.data * np.log(self.data) * out.grad
            
            out._backward = _backward
            return out
        else:
            # Handle Value ** constant case
            assert isinstance(other, (int, float)), "power must be a number or Value"
            out = Value(self.data ** other, _children=(self,), _op=f'**{other}')

            def _backward():
                self.grad += other * (self.data ** (other - 1)) * out.grad

            out._backward = _backward
            return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def main():
    # Example 1: Your requested expression f = ((a*a) + b) ** c
    print("=== Simple Expression: f = ((a*a) + b) ** c ===")
    a, b, c = Value(2.0), Value(3.0), Value(2.0)
    
    # Forward pass
    start = time.time()
    f = ((a * a) + b) ** c
    fw_time = time.time() - start
    
    # Backward pass  
    start = time.time()
    f.backward()
    bw_time = time.time() - start
    
    print(f"Result: {f.data}")
    print(f"Times: Forward {fw_time*1000:.3f}ms, Backward {bw_time*1000:.3f}ms")
    print(f"Gradients: a={a.grad:.3f}, b={b.grad:.3f}, c={c.grad:.3f}")
    print()

    # Example 2: Small Neural Network
    print("=== Small Neural Network ===")
    mlp = MLP(3, [4, 1])  # Simplified: 3->4->1
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    
    # Forward pass
    start = time.time()
    output = mlp(x)
    loss = (output - Value(1.0)) ** 2  # Loss
    fw_time = time.time() - start
    
    # Backward pass
    start = time.time()
    loss.backward()
    bw_time = time.time() - start
    
    print(f"Loss: {loss.data:.4f}")
    print(f"Times: Forward {fw_time*1000:.3f}ms, Backward {bw_time*1000:.3f}ms")
    print(f"Parameters: {len(mlp.parameters())}")
    print()

    # Example 3: Larger Network Benchmark
    print("=== Larger Network Benchmark ===")
    big_mlp = MLP(5, [20, 10, 1])  # 5->20->10->1
    x_big = [Value(np.random.randn()) for _ in range(5)]
    
    # Forward pass
    start = time.time()
    output_big = big_mlp(x_big)
    loss_big = (output_big - Value(0.0)) ** 2
    fw_time = time.time() - start
    
    # Backward pass
    start = time.time()
    loss_big.backward()
    bw_time = time.time() - start
    
    print(f"Loss: {loss_big.data:.6f}")
    print(f"Times: Forward {fw_time*1000:.3f}ms, Backward {bw_time*1000:.3f}ms")
    print(f"Parameters: {len(big_mlp.parameters())}")
    total_time = (fw_time + bw_time) * 1000
    print(f"Total: {total_time:.3f}ms")


if __name__ == "__main__":
    main()
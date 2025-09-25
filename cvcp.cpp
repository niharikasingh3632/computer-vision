#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>

// Forward declaration to allow ValuePtr to be used inside the Value class.
class Value;

// Use a shared pointer for automatic memory management,
// which is a good C++ equivalent for Python's object-oriented behavior.
using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::string _op;
    std::string _label;
    
    // A set of shared pointers to represent the "children" or previous nodes
    // in the computational graph.
    std::set<ValuePtr> _prev;

    // A function object to store the logic for the backward pass.
    std::function<void()> _backward;

    // Constructor to initialize the Value object.
    Value(double data, const std::set<ValuePtr>& children = {}, const std::string& op = "", const std::string& label = "")
        : data(data), grad(0.0), _op(op), _label(label), _prev(children), _backward([](){}) {}

    // Overload the << operator to allow easy printing of a Value object.
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        return os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
    }

    // Power operator
    ValuePtr pow(double other) {
        auto out = std::make_shared<Value>(std::pow(data, other), std::set<ValuePtr>{shared_from_this()}, "**" + std::to_string(other));
        out->_backward = [this, other, out](){
            this->grad += other * std::pow(this->data, other - 1) * out->grad;
        };
        return out;
    }

    // Hyperbolic tangent activation function
    ValuePtr tanh() {
        double t = std::tanh(data);
        auto out = std::make_shared<Value>(t, std::set<ValuePtr>{shared_from_this()}, "tanh");
        out->_backward = [this, t, out](){
            this->grad += (1 - t*t) * out->grad;
        };
        return out;
    }

    // Exponential function
    ValuePtr exp() {
        double out_data = std::exp(data);
        auto out = std::make_shared<Value>(out_data, std::set<ValuePtr>{shared_from_this()}, "exp");
        out->_backward = [this, out_data, out](){
            this->grad += out_data * out->grad;
        };
        return out;
    }

    // The core backward propagation function.
    void backward() {
        std::vector<ValuePtr> topo;
        std::set<ValuePtr> visited;
        std::function<void(ValuePtr)> build_topo = 
            [&](ValuePtr v){
            if (visited.find(v) == visited.end()){
                visited.insert(v);
                for(const auto& child : v->_prev){
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };

        build_topo(shared_from_this());

        this->grad = 1.0;
        std::reverse(topo.begin(), topo.end());
        for(const auto& node : topo){
            node->_backward();
        }
    }
};

// Global helper functions to handle operator overloading with shared pointers
ValuePtr operator+(const ValuePtr& lhs, const ValuePtr& rhs) { 
    auto out = std::make_shared<Value>(lhs->data + rhs->data, std::set<ValuePtr>{lhs, rhs}, "+");
    out->_backward = [lhs, rhs, out](){
        lhs->grad += 1.0 * out->grad;
        rhs->grad += 1.0 * out->grad;
    };
    return out;
}

ValuePtr operator+(const ValuePtr& lhs, double rhs) { return lhs + std::make_shared<Value>(rhs); }
ValuePtr operator+(double lhs, const ValuePtr& rhs) { return std::make_shared<Value>(lhs) + rhs; }

ValuePtr operator*(const ValuePtr& lhs, const ValuePtr& rhs) { 
    auto out = std::make_shared<Value>(lhs->data * rhs->data, std::set<ValuePtr>{lhs, rhs}, "*");
    out->_backward = [lhs, rhs, out](){
        lhs->grad += rhs->data * out->grad;
        rhs->grad += lhs->data * out->grad;
    };
    return out;
}

ValuePtr operator*(const ValuePtr& lhs, double rhs) { return lhs * std::make_shared<Value>(rhs); }
ValuePtr operator*(double lhs, const ValuePtr& rhs) { return std::make_shared<Value>(lhs) * rhs; }

ValuePtr operator-(const ValuePtr& lhs, const ValuePtr& rhs) { 
    auto out = std::make_shared<Value>(lhs->data - rhs->data, std::set<ValuePtr>{lhs, rhs}, "-");
    out->_backward = [lhs, rhs, out](){
        lhs->grad += 1.0 * out->grad;
        rhs->grad += -1.0 * out->grad;
    };
    return out;
}

ValuePtr operator-(const ValuePtr& lhs, double rhs) { return lhs - std::make_shared<Value>(rhs); }
ValuePtr operator-(double lhs, const ValuePtr& rhs) { return std::make_shared<Value>(lhs) - rhs; }

ValuePtr operator-(const ValuePtr& v) { return std::make_shared<Value>(0.0) - v; }

ValuePtr operator/(const ValuePtr& lhs, const ValuePtr& rhs) { return lhs * rhs->pow(-1); }
ValuePtr operator/(const ValuePtr& lhs, double rhs) { return lhs / std::make_shared<Value>(rhs); }
ValuePtr operator/(double lhs, const ValuePtr& rhs) { return std::make_shared<Value>(lhs) / rhs; }

class Neuron {
public:
    std::vector<ValuePtr> w;
    ValuePtr b;
    
    Neuron(int nin) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        for(int i = 0; i < nin; ++i) {
            w.push_back(std::make_shared<Value>(dis(gen)));
        }
        b = std::make_shared<Value>(dis(gen));
    }

    ValuePtr operator()(const std::vector<ValuePtr>& x) {
        ValuePtr act = b;
        for(size_t i = 0; i < w.size(); ++i) {
            act = act + w[i] * x[i];
        }
        return act->tanh();
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> params = w;
        params.push_back(b);
        return params;
    }
};

class Layer {
public:
    std::vector<Neuron> neurons;
    Layer(int nin, int nout) {
        for(int i = 0; i < nout; ++i) {
            neurons.emplace_back(nin);
        }
    }

    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x) {
        std::vector<ValuePtr> outs;
        for(auto& neuron : neurons) {
            outs.push_back(neuron(x));
        }
        return outs;
    }

    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> params;
        for(auto& neuron : neurons) {
            std::vector<ValuePtr> neuron_params = neuron.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};

class MLP {
public:
    std::vector<Layer> layers;
    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        for(size_t i = 0; i < nouts.size(); ++i) {
            layers.emplace_back(sz[i], sz[i+1]);
        }
    }

    ValuePtr operator()(const std::vector<ValuePtr>& x) {
        std::vector<ValuePtr> current_x = x;
        for(size_t i = 0; i < layers.size() - 1; ++i) {
            current_x = layers[i](current_x);
        }
        return layers.back()(current_x)[0];
    }
    
    std::vector<ValuePtr> parameters() {
        std::vector<ValuePtr> params;
        for(auto& layer : layers) {
            std::vector<ValuePtr> layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

void simpleExpressionExample() {
    std::cout << "=== Simple Expression: f = ((a*a) + b) ** c ===" << std::endl;
    auto a = std::make_shared<Value>(2.0);
    auto b = std::make_shared<Value>(3.0);
    auto c = std::make_shared<Value>(2.0);
    
    // Forward pass with timing
    auto start_fw = std::chrono::high_resolution_clock::now();
    auto f = (a * a + b)->pow(c->data);
    auto end_fw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fw_time = end_fw - start_fw;

    // Backward pass with timing
    auto start_bw = std::chrono::high_resolution_clock::now();
    f->backward();
    auto end_bw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> bw_time = end_bw - start_bw;
    
    std::cout << "Result: " << f->data << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Times: Forward " << fw_time.count() << "ms, Backward " << bw_time.count() << "ms" << std::endl;
    std::cout << "Gradients: a=" << a->grad << ", b=" << b->grad << ", c=" << c->grad << std::endl;
    std::cout << std::endl;
}

void smallNeuralNetworkExample() {
    std::cout << "=== Small Neural Network ===" << std::endl;
    MLP mlp(3, {4, 1}); // Simplified: 3->4->1
    std::vector<ValuePtr> x = {std::make_shared<Value>(1.0), std::make_shared<Value>(-2.0), std::make_shared<Value>(3.0)};
    
    // Forward pass with timing
    auto start_fw = std::chrono::high_resolution_clock::now();
    auto output = mlp(x);
    auto loss = output - std::make_shared<Value>(1.0)->pow(2);
    auto end_fw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fw_time = end_fw - start_fw;
    
    // Backward pass with timing
    auto start_bw = std::chrono::high_resolution_clock::now();
    loss->backward();
    auto end_bw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> bw_time = end_bw - start_bw;
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Loss: " << loss->data << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Times: Forward " << fw_time.count() << "ms, Backward " << bw_time.count() << "ms" << std::endl;
    std::cout << "Parameters: " << mlp.parameters().size() << std::endl;
    std::cout << std::endl;
}

void largeNeuralNetworkExample() {
    std::cout << "=== Larger Network Benchmark ===" << std::endl;
    MLP big_mlp(5, {20, 10, 1}); // 5->20->10->1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    std::vector<ValuePtr> x_big;
    for(int i = 0; i < 5; ++i) {
        x_big.push_back(std::make_shared<Value>(d(gen)));
    }
    
    // Forward pass with timing
    auto start_fw = std::chrono::high_resolution_clock::now();
    auto output_big = big_mlp(x_big);
    auto loss_big = output_big - std::make_shared<Value>(0.0)->pow(2);
    auto end_fw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fw_time = end_fw - start_fw;
    
    // Backward pass with timing
    auto start_bw = std::chrono::high_resolution_clock::now();
    loss_big->backward();
    auto end_bw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> bw_time = end_bw - start_bw;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Loss: " << loss_big->data << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Times: Forward " << fw_time.count() << "ms, Backward " << bw_time.count() << "ms" << std::endl;
    std::cout << "Parameters: " << big_mlp.parameters().size() << std::endl;
    double total_time = (fw_time.count() + bw_time.count());
    std::cout << "Total: " << total_time << "ms" << std::endl;
}

int main() {
    simpleExpressionExample();
    smallNeuralNetworkExample();
    largeNeuralNetworkExample();
    return 0;
}

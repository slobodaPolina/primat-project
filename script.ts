// Эта часть кода отвечает за градиентный спуск

// returns if a and b are equal to within eps
function similar(a : number, b : number, eps1 : number) {
    return Math.abs(a - b) < eps1;
}

// count lambda to minimalize f(x + l * grad(x))
function countLambda(x : number) {
    let eps = 0.000001;
    let a = -1000.0, b = 1000.0; // the borders
    let d = eps / 4.0;
    while (Math.abs(a - b) > eps) {
        let l1 = (a + b) / 2.0 - d;
        let l2 = (a + b) / 2.0 + d;
        if (f(x - (grad(x) * l1)) >= f(x - (grad(x) * l2))) {
            a = l1;
        }
        if (f(x - (grad(x) * l1)) <= f(x - (grad(x) * l2))) {
            b = l2;
        }
    }
    return (a + b) / 2.0;
}

function findMin() { // find minimum of f(x)
    let lambda = 0;
    let ef = 0.001;
    let eps1 = 0.001;
    let next = -1000.0;
    let previous = next;
    do {
        previous = next;
        lambda = countLambda(previous);
        next = previous - (grad(previous) * lambda);
        console.log("Next vector is " + next + "." + "Value is " + f(next));
    } while (Math.abs(f(previous) - f(next)) > ef && !similar(previous, next, eps1));
    let res = (previous + next) / 2.0;
    console.log(
        "----------The minimum is found at (" + res + ") and f is " + f(res)
    );
    return res;
}

// минимизируемая функция
function f(x : number) {
    return 0.5 * Math.pow((run(x) - modeling(x)), 2);
}

// градиент от f(x)
function grad(x : number) {
    return x;
}

// -----------------------------------------------------------------------

function modeling(x : number) {
    return Math.log(1 + Math.pow(Math.E, x));
 }

// выходная вершина дает ответ
// и допустим 2 скрытых слоя по 10 нейронов.
// в каждый из 10 входит х, там будет вес,
// получится 10 весов на первый скрытый слой + 10 * 10 весов на второй + 10 на финальный (внешний)

var NODES = 10;

var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида

var weights = Array((NODES + 2) * NODES).fill(0);

// x is the input. here we just normalize it (make it from 0 to 1)
// границы х от -10 до +10
var firstNode = (x : number) => (x + 10.0) / 20.0;

// x is normalized result of the first layer.
// number is the number of the node in the row of the second nodes
function secondNode(x : number, number : number) {
    return activation(weights[number] * x);
}

// results is an array of 10 results of the second layer
function thirdNode(results : number[], number : number) {
    var sum = 0;
    results.forEach((index, result) => sum += result * weights[NODES + number * NODES + index]);
    return activation(sum);
}

// an external node (final one). Returns the prediction
function externalNode(results : number[]) {
    return thirdNode(results, NODES);
}

// the main function
function run(parameter : number) {
    let frt = firstNode(parameter);
    let seconds = [];
    for(let i = 0; i < NODES; i++) {
        seconds.push(secondNode(frt, i));
    }

    let thirds = [];
    for(let i = 0; i < NODES; i++) {
        thirds.push(thirdNode(seconds, i));
    }

    let result = externalNode(thirds);
    console.log("function of " + parameter + " returned " + result +
        ". Thereas real value is " + modeling(parameter)
    );
    return result;
}

// Эта часть кода отвечает за градиентный спуск
/*
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
    } while (
        Math.abs(f(previous) - f(next)) > ef && !similar(previous, next, eps1)
    );
    let res = (previous + next) / 2.0;
    console.log(
        "----------The minimum is found at (" + res + ") and f is " + f(res)
    );
    return res;
}
*/
// минимизируемая функция
function f(x : number) {
    return 0.5 * Math.pow((runAll(x) - modeling(x)), 2);
}

// градиент от функции протерь
function grad(x : number) {
    // вектор частных производных для каждой связи между нейронами
    let res = Array(NODES * (NODES + 2));
    for(let k = 0; k < res.length; k++) {
        // этот вес из нейрона i в j
        let previous; // выход предыдущего нейрона
        let der; // производная функции возбуждения по ее аргументу в текущем нейроне
        if (k < NODES) {
            // перыдущий нейрон - начальный, ведет во второй слой
            previous = firstNode(x);
            // последующий нейрон - нейрон второго слоя
            der = secondNodeDer(previous, k);
        } else if (k < NODES * (NODES + 1)) {
            // ведет со второго на третьий слой
            previous = secondNode(firstNode(x), Math.floor((k - NODES) / NODES));

            let seconds = run12(x);
            der = thirdNodeDer(seconds, Math.floor((k - NODES) / NODES));
        } else {
            // с третьего во внешний
            let seconds = run12(x);
            previous = thirdNode(seconds, k - (NODES * (NODES + 1)));
            let thirds = run123(x);
            der = externalNodeDer(thirds);
        }
        res[k] = previous * der;
    }
    // теперь посчитаем ошибки нейронов путем алгоритма обратного распространения ошибки
    // ошибка на выходном
    let finalMistake = (modeling(x) - runAll(x)) * externalNodeDer(run123(x));
    let mistakes = Array(2 * NODES + 2).fill(0);
    mistakes[2 * NODES + 1] = finalMistake;
    for (let i = 2 * NODES; i >= 0; i--) {
        countMistake(mistakes, i, x);
    }
    let gradVector = Array(NODES * (NODES + 2)).fill(0);
    for (let i = 0; i < gradVector.length; i++) {
        gradVector[i] = res[i] * mistakes[parseDestinationByWeightNumber(i)];
    }
    return gradVector;
}

// ошибка нейрона номер NodeNumber при запуске сети на числе x
function countMistake(mistakes : number[], NodeNumber : number, x : number) {
    let result = 0;
    if (NodeNumber >= NODES + 1) { // вершина 3 слоя
        result = mistakes[2 * NODES + 1] *
            links[NodeNumber][2 * NODES + 1] *
            thirdNodeDer(run12(x), (NodeNumber - NODES - 1));
    } else if(NodeNumber >= 1) { // вершина 2 слоя
        let der = secondNodeDer(firstNode(x), (NodeNumber - 1));
        for(let j = NODES + 1; j < (2 * NODES + 1); j++) { // бежим по 3 слою
            result += der * mistakes[j] * links[NodeNumber][j];
        }
    } else { // вершина 1 слоя (входная)
        let der = firstNodeDer(x);
        for(let j = 1; j < (NODES + 1); j++) { // бежим по 2 слою
            result += der * mistakes[j] * links[NodeNumber][j];
        }
    }
    mistakes[NodeNumber] = result;
}

// меняет веса, prevDeltas, prevResult и nu на каждом этапе обучения
function changeWeights (x : number, prevDeltas : number[], prevResult : number, nu : number) {
    let p = 1; // коэффициент регуляризации
    let mu = 1; // коэффициент момента
    let newResult = runAll(x);
    let delta = newResult - 1.01 * prevResult;
    prevResult = newResult;
    let gradVector = grad(x);
    for(let i = 0; i < links.length; i++) {
        for(let j = 0; j < links.length; j++) {
            if(links[i][j] != undefined) {
                prevDeltas[i][j] = nu * (gradVector[getWeightNumberByNodes(i, j)] + p * links[i][j]) + mu * prevDeltas[i][j];
                links[i][j] -= prevDeltas[i][j];
            }
        }
    }
    nu = delta > 0 ? nu * 0.99 : nu * 1.01;
}

// получить номер узла, в который седет связь, по номеру связи
function parseDestinationByWeightNumber(i : number) {
    return i < NODES ?
        i + 1 :
            i < (NODES + 1) * NODES ?
                NODES + 1 + Math.floor(i / NODES) :
                NODES * (NODES + 2) - 1;
}

// получить номер вершины, из которой ведет связь
function parseSourceByWeightNumber(i : number) {
    return i < NODES ? // связь из 1 во 2 слой
            0 :
            i < (NODES + 1) * NODES ? // из 2 в 3
                1 + (i % NODES) :
                1 + NODES + (i % NODES); // из 3 в 4
}

function getWeightNumberByNodes(from : number, to : number) {
    if (from === 0) { // 1 to 2
        return to - 1;
    } else if (from <= NODES) { // 2 to 3
        return NODES + (to - 1 - NODES) * NODES + from - 1;
    } else { // 3 to 4
        return (NODES + 1) * NODES + (from - 1 - NODES);
    }
}

// -----------------------------------------------------------------------

function modeling(x : number) {
    return Math.log(1 + Math.pow(Math.E, x));
 }

// выходная вершина дает ответ
// и допустим 2 скрытых слоя по 10 нейронов.
// в каждый из 10 входит х, там будет вес,
// получится 10 весов на первый скрытый слой
// + 10 * 10 весов на второй + 10 на финальный (внешний)

var NODES = 10; // количество вершин в каждом внутреннем слое
var links = Array(NODES * 2 + 2); // двумерный массив связей между нейронами
for (let i = 0; i < links.length; i++) {
    links[i] = Array(NODES * 2 + 2);
}

var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида
var activationDerivate = (x : number) => Math.pow(Math.E, -x) / Math.pow(Math.pow(Math.E, -x) + 1, 2);

// x is the input. here we just normalize it (make it from 0 to 1)
// границы х от -10 до +10
var firstNode = (x : number) => activation((x + 10.0) / 20.0);

// x is normalized result of the first layer.
// number is the number of the node in the row of the second nodes
function secondNode(x : number, number : number) {
    return activation(links[0][number + 1] * x);
}

// results is an array of 10 results of the second layer
//number is still number of the node of 3st layer
function thirdNode(results : number[], number : number) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + index][NODES + 1 + number]
    );
    return activation(sum);
}

// an external node (final one). Returns the prediction
function externalNode(results : number[]) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + NODES + index][2 * NODES + 1]
    );
    return activation(sum);
}


// they return just derivates of the arguments given
var firstNodeDer = (x : number) => activationDerivate((x + 10.0) / 20.0);

function secondNodeDer(x : number, number : number) {
    return activationDerivate(links[0][number + 1] * x);
}

function thirdNodeDer(results : number[], number : number) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + index][NODES + 1 + number]
    );
    return activationDerivate(sum);
}

function externalNodeDer(results : number[]) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + NODES + index][2 * NODES + 1]
    );
    return activationDerivate(sum);
}

function run12 (x : number) {
    let frt = firstNode(x);
    let seconds = [];
    for(let i = 0; i < NODES; i++) {
        seconds.push(secondNode(frt, i));
    }
    return seconds;
}

function run123 (x : number) {
    let seconds = run12(x);
    let thirds = [];
    for(let i = 0; i < NODES; i++) {
        thirds.push(thirdNode(seconds, i));
    }
    return thirds;
}

// the main function
function runAll(x : number) {
    let thirds = run123(x);
    let result = externalNode(thirds);
    console.log("function of " + x + " returned " + result +
        ". Thereas real value is " + modeling(x)
    );
    return result;
}

for(let i = 0; i < links.length; i++) {
    for (let j = 0; j < links.length; j++) {
        if ((i === 0 && j > 0 && j < NODES + 1) || // из 1 слоя во второй
            // из второго в третий
            (i > 0 && i < NODES + 1 && j >= NODES + 1 && j <= 2 * NODES) ||
            // из третьего в четвертый
            (i >= NODES + 1 && i <= 2 * NODES && j === 2 * NODES + 1)
        ) {
            links[i][j] = 0;
        } else {
            links[i][j] = undefined;
        }
    }
}
runAll(0);

import * as fs from 'fs';

function modeling(x : number) { // функция, которую моделируем
    return Math.log(1 + Math.pow(Math.E, x));
 }

// выходная вершина дает ответ
// и допустим 2 скрытых слоя по 10 нейронов.
// в каждый из 10 входит х, там будет вес,
// получится 10 весов на первый скрытый слой
// + 10 * 10 весов на второй + 10 на финальный (внешний)

var NODES = 10; // количество вершин в каждом внутреннем слое
var length = NODES * 2 + 2; // всего вершин. Начальная, конечная и 2 слоя между ними по NODES
var links = Array(length); // двумерный массив связей между нейронами
// изначально заполним весы нулями
for(let i = 0; i < length; i++) {
    links[i] = Array(length);
    for (let j = 0; j < length; j++) {
        links[i][j] = getLayerOfNode(j) - getLayerOfNode(i) === 1 ?
            0 : undefined;
    }
}

// градиент от функции протерь
function grad(x : number) {
    // вектор частных производных для каждой связи между нейронами
    let gradVector = Array(length);
    for(let k = 0; k < length; k++) {
        gradVector[k] = Array(length);
        for (let r = 0; r < length; r++) {
            if(links[k][r] != undefined) {
                // этот вес из нейрона k в r
                let previous: number; // выход предыдущего нейрона
                let der: number; // производная функции возбуждения по ее аргументу в текущем нейроне
                if (getLayerOfNode(k) === 1) {
                    // перыдущий нейрон - начальный, ведет во второй слой
                    previous = firstNode(x);
                    // последующий нейрон - нейрон второго слоя
                    der = secondNodeDer(previous, r);
                } else if (getLayerOfNode(k) === 2) {
                    // ведет со второго на третьий слой
                    previous = secondNode(firstNode(x), k);
                    let seconds = run12(x);
                    der = thirdNodeDer(seconds, r);
                } else {
                    // с третьего во внешний
                    let seconds = run12(x);
                    previous = thirdNode(seconds, k);
                    let thirds = run123(x);
                    der = externalNodeDer(thirds);
                }
                gradVector[k][r] = previous * der;
            }
        }
    }
    // теперь посчитаем ошибки нейронов путем алгоритма обратного распространения ошибки
    // ошибка на выходном
    let finalMistake = (modeling(x) - runAll(x)) * externalNodeDer(run123(x));
    let mistakes = Array(length).fill(0); // для всех вершинок
    mistakes[length - 1] = finalMistake;
    for (let i = length - 2; i >= 0; i--) {
        countMistake(mistakes, i, x);
    }
    for (let i = 0; i < length; i++) {
        for (let j = 0; j < length; j++) {
            if (gradVector[i][j] != undefined) {
                gradVector[i][j] *= mistakes[j];
            }
        }
    }
    return gradVector;
}

function getLayerOfNode(NodeNumber : number) {
    return NodeNumber === 0 ?
        1 :
        NodeNumber > 0 && NodeNumber <= NODES ?
            2 :
            NodeNumber > NODES && NodeNumber <= 2 * NODES ?
                3 :
                4;
}

// ошибка нейрона номер NodeNumber при запуске сети на числе x
function countMistake(mistakes : number[], NodeNumber : number, x : number) {
    let result = 0;
    if (getLayerOfNode(NodeNumber) === 3) { // вершина 3 слоя
        result = mistakes[length - 1] *
            links[NodeNumber][length - 1] *
            thirdNodeDer(run12(x), NodeNumber);
    } else if(getLayerOfNode(NodeNumber) === 2) { // вершина 2 слоя
        let der = secondNodeDer(firstNode(x), NodeNumber);
        for(let j = NODES + 1; j < length - 1; j++) { // бежим по 3 слою
            result += der * mistakes[j] * links[NodeNumber][j];
        }
    } else if (getLayerOfNode(NodeNumber) === 1){ // вершина 1 слоя (входная)
        let der = firstNodeDer(x);
        for(let j = 1; j < (NODES + 1); j++) { // бежим по 2 слою
            result += der * mistakes[j] * links[NodeNumber][j];
        }
    } else {
        console.error("IMPOSSIBLE LAYER NUMBER");
    }
    mistakes[NodeNumber] = result;
}

var prevDeltas = Array(length);
for(let i = 0; i < length; i++) {
    prevDeltas[i] = Array(length).fill(0);
}
var nu = 1;
// меняет веса, prevDeltas, prevResult и nu на каждом этапе обучения.
// вернет prevResult
function changeWeights (x : number, prevResult : number) {
    let p = 1; // коэффициент регуляризации
    let mu = 1; // коэффициент момента
    let newResult = runAll(x);
    let delta = newResult - 1.01 * prevResult;
    prevResult = newResult;
    let gradVector = grad(x);
    for(let i = 0; i < length; i++) {
        for(let j = 0; j < length; j++) {
            if(links[i][j] != undefined) {
                prevDeltas[i][j] =
                    nu * (gradVector[i][j] + p * links[i][j]) +
                        mu * prevDeltas[i][j];
                links[i][j] -= prevDeltas[i][j];
            }
        }
    }
    console.log("Changing weights. Deltas of the weights are : ");
    for(let i = 0; i < length; i++) {
        for(let j = 0; j < length; j++) {
            if(links[i][j] != undefined) {
                console.log("minused " + prevDeltas[i][j] + " to " + links[i][j]);
            }
        }
    }

    nu = delta > 0 ? nu * 0.99 : nu * 1.01;
    return prevResult;
}

// -----------------------------------------------------------------------

var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида
var activationDerivate = (x : number) => Math.pow(Math.E, -x) / Math.pow(Math.pow(Math.E, -x) + 1, 2);

// x is the input. here we just normalize it (make it from 0 to 1)
// границы х от -10 до +10
var firstNode = (x : number) => activation((x + 10.0) / 20.0);

// x is normalized result of the first layer.
// NodeNumber is the number of the node in the list of all nodes
function secondNode(x : number, NodeNumber : number) {
    return activation(links[0][NodeNumber] * x);
}

// results is an array of 10 results of the second layer
// NodeNumber is still the number of the node in the full list
function thirdNode(results : number[], NodeNumber : number) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + index][NodeNumber]
    );
    return activation(sum);
}

// an external node (final one). Returns the prediction
function externalNode(results : number[]) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + NODES + index][length - 1]
    );
    return activation(sum);
}

// they return just derivates of the arguments given
var firstNodeDer = (x : number) => activationDerivate((x + 10.0) / 20.0);

function secondNodeDer(x : number, NodeNumber : number) {
    return activationDerivate(links[0][NodeNumber] * x);
}

function thirdNodeDer(results : number[], NodeNumber : number) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + index][NodeNumber]
    );
    return activationDerivate(sum);
}

function externalNodeDer(results : number[]) {
    var sum = 0;
    results.forEach(
        (result, index) => sum += result * links[1 + NODES + index][length - 1]
    );
    return activationDerivate(sum);
}

function run12 (x : number) {
    let frt = firstNode(x);
    let seconds = [];
    for(let i = 1; i < NODES + 1; i++) {
        seconds.push(secondNode(frt, i));
    }
    return seconds;
}

function run123 (x : number) {
    let seconds = run12(x);
    let thirds = [];
    for(let i = NODES + 1; i < length - 1; i++) {
        thirds.push(thirdNode(seconds, i));
    }
    return thirds;
}

// the main function
function runAll(x : number) {
    let thirds = run123(x);
    let result = externalNode(thirds);
    return result * 10; // тут надо обратно как-то вмасштабироваться, на выходе от 0 до 10
}

let prevResult = 0;
let x: number;
let data = [0, 0, 0, 0, 0];
//let data = fs.readFileSync('data.txt', { encoding: 'utf-8' }).split('\n');
data.forEach(s => {
    x = Number(s); // для каждой точки из файла запускаем и корректируем, если нужно
    let result = runAll(x);
    let realValue = modeling(x);
    if (Math.abs(result - realValue) > 0.001) {
        console.log("function of " + x + " returned " + result);
        console.log("mistake is " + Math.abs(result - realValue));
        prevResult = changeWeights(x, prevResult);
    } else {
        console.log("OK");
    }
    console.log("-----------------------------------");
});

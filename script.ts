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
            Math.random() : undefined; // связываем вершины соседних слоев, генерируем случайное значение.
            // если связи нет, оставляем undefined
    }
}

function getLayerOfNode(NodeNumber : number) { // получаем номер слоя, к которому принадлежит вершина
    return NodeNumber === 0 ?
        1 :
        NodeNumber > 0 && NodeNumber <= NODES ?
            2 :
            NodeNumber > NODES && NodeNumber <= 2 * NODES ?
                3 :
                4;
}

// ------------------------------------------------------------------------------------------------------------

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
                    previous = firstNode(x, activation);
                    // последующий нейрон - нейрон второго слоя
                    der = secondNode(previous, r, activationDerivate);
                } else if (getLayerOfNode(k) === 2) {
                    // ведет со второго на третьий слой
                    previous = secondNode(firstNode(x, activation), k, activation);
                    let seconds = run12(x);
                    der = thirdNode(seconds, r, activationDerivate);
                } else {
                    // с третьего во внешний
                    let seconds = run12(x);
                    previous = thirdNode(seconds, k, activation);
                    let thirds = run123(x);
                    der = externalNode(thirds, activationDerivate);
                }
                gradVector[k][r] = previous * der;
            }
        }
    }
    // теперь посчитаем ошибки нейронов путем алгоритма обратного распространения ошибки
    // ошибка на выходном
    let finalMistake = (modeling(x) - runAll(x)) * externalNode(run123(x), activationDerivate);
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

// ошибка нейрона номер NodeNumber при запуске сети на числе x
function countMistake(mistakes : number[], NodeNumber : number, x : number) {
    let result = 0;
    if (getLayerOfNode(NodeNumber) === 3) { // вершина 3 слоя
        result = mistakes[length - 1] *
            links[NodeNumber][length - 1] *
            thirdNode(run12(x), NodeNumber, activationDerivate);
    } else if(getLayerOfNode(NodeNumber) === 2) { // вершина 2 слоя
        let der = secondNode(firstNode(x, activation), NodeNumber, activationDerivate);
        for(let j = NODES + 1; j < length - 1; j++) { // бежим по 3 слою
            result += der * mistakes[j] * links[NodeNumber][j];
        }
    } else if (getLayerOfNode(NodeNumber) === 1){ // вершина 1 слоя (входная)
        let der = firstNode(x, activationDerivate);
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
var n = 0.5;
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
                    n * (gradVector[i][j] + p * links[i][j]) +
                        mu * prevDeltas[i][j];
                links[i][j] -= prevDeltas[i][j];
            }
        }
    }
    console.log("Changing weights. Deltas of the weights are : ");
    for(let i = 0; i < length; i++) {
        for(let j = 0; j < length; j++) {
            if(links[i][j] != undefined) {
                console.log("minused " + prevDeltas[i][j] + " from " + links[i][j]);
            }
        }
    }

    n = delta > 0 ? n * 0.99 : n * 1.01;
    return prevResult;
}

// ------------------------------------------------------------------------------------------------------

// функция активации
var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида

// производная функции активации
var activationDerivate = (x : number) =>
        Math.pow(Math.E, -x) / Math.pow(Math.pow(Math.E, -x) + 1, 2);

// x - число, подаваемое сети на вход. Нормализуем его.
// границы х от -10 до +10 => от 0 до 1
// f - функция, которую выполнить - активация или ее производная
var firstNode = (x : number, f : (n : number) => number) => f((x + 10.0) / 20.0);

// NodeNumber - номер вершины среди всех вершинок. х - выход первого слоя
function secondNode(x : number, NodeNumber : number, f : (n : number) => number) {
    if (getLayerOfNode(NodeNumber) !== 2) { // на всякий случай
        console.error("GOT NODE NOT OF 2 LAYER");
    }
    if (links[0][NodeNumber] == undefined) {
        console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
    }
    return f(links[0][NodeNumber] * x);
}

// results is an array of 10 results of the second layer
// NodeNumber is still the number of the node in the full list
function thirdNode(results : number[], NodeNumber : number, f : (n : number) => number) {
    if (getLayerOfNode(NodeNumber) !== 3) { // на всякий случай
        console.error("GOT NODE NOT OF 3 LAYER");
    }

    var sum = 0;
    results.forEach( // все выходы 2 слоя домножаем на веса и суммируем
        (result, index) => {
            if (links[1 + index][NodeNumber] == undefined) {
                console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
            }
            sum += result * links[1 + index][NodeNumber];
        }
    );
    return f(sum);
}

// an external node (final one). Returns the prediction
function externalNode(results : number[], f : (n : number) => number) {
    var sum = 0;
    results.forEach(
        (result, index) => {
            if (links[1 + NODES + index][length - 1] == undefined) {
                console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
            }
            // аналогично перебираем 3 слой,
            // домножаем на веса из вершины 3 слоя в последнюю (вершину 4 слоя)
            sum += result * links[1 + NODES + index][length - 1];
        }
    );
    return f(sum);
}

function run12 (x : number) { // запустить 1 и 2 слой сети
    let frt = firstNode(x, activation);
    let seconds = [];
    for(let i = 1; i < NODES + 1; i++) {
        seconds.push(secondNode(frt, i, activation));
    }
    return seconds;
}

function run123 (x : number) { // запуск 1, 2 и 3 слоев
    let seconds = run12(x);
    let thirds = [];
    for(let i = NODES + 1; i < length - 1; i++) {
        thirds.push(thirdNode(seconds, i, activation));
    }
    return thirds;
}

// запустить все слои (запустить сеть)
function runAll(x : number) {
    let thirds = run123(x);
    let result = externalNode(thirds, activation);
    // финальный слой отдает результат активации, надо его как-то отмасштабировать
    return result * 10;
}

let prevResult = 0;
let x: number;
let data = [0, 0, 0, 0, 0]; // это я пока взяла для теста и отладки
//let data = fs.readFileSync('data.txt', { encoding: 'utf-8' }).split('\n');
data.forEach(s => {
    x = Number(s); // для каждой точки из файла запускаем и корректируем, если нужно
    let result = runAll(x);
    let realValue = modeling(x);
    if (Math.abs(result - realValue) > 0.001) { // если сеть отвечает совсем не так, как надо, обучаем
        console.log("function of " + x + " returned " + result);
        console.log("mistake is " + Math.abs(result - realValue));
        prevResult = changeWeights(x, prevResult);
    } else {
        console.log("OK");
    }
    console.log("-----------------------------------");
});

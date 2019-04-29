import * as fs from 'fs';

// выдает значения от 0 до 10
var abs_modeling = (x: number) => Math.log(1 + Math.pow(Math.E, x)); // моделируемая функция

function modeling(x : number) { // идеальная активация нейрона на выходе от 0 до 1
    // Нормализуем идеальный результат, теперь он по идее должен стать идеальной степенью активации
    return (abs_modeling(x) - abs_modeling(-10)) /
                (abs_modeling(10.0) - abs_modeling(-10));
 }

// по значению уровня активации восстановит ответ в абсолютных цифрах
 function renormalize(x: number) {
     return ((abs_modeling(10.0) - abs_modeling(-10)) * x) + abs_modeling(-10);
 }

// выходная вершина дает ответ
// и допустим 2 скрытых слоя по 10 нейронов.
// в каждый из 10 входит х, там будет вес,
// получится 10 весов на первый скрытый слой
// + 10 * 10 весов на второй + 10 на финальный (внешний)

var NODES = 20; // количество вершин в каждом внутреннем слое
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

var n = 1; // скорость обучения, нужно подбирать

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

// меняет веса и n на каждом этапе обучения
function changeWeights (x : number) {
    let deltas = Array(length);
    for(let i = 0; i < length; i++) {
        deltas[i] = Array(length);
    }
    let result = runAll(x);
    let error = modeling(x) - result;
    deltas[length - 1] = activationDerivate(result) * error; // насколько изменить выходной результат
    //console.log(result);
    //console.log(activationDerivate(result));
    //console.log(deltas[length - 1]);

    // обработаем 3 слой
    for(let i = 1 + NODES; i < length - 1; i++) {
        if(links[i][length - 1] != undefined) {
            deltas[i] = activationDerivate(thirdNode(run12(x), i)) *
                            deltas[length - 1] * links[i][length - 1];
            //console.log(deltas[i]);
        } else {
            console.error("NO WEIGHT BUT HAS TO BE");
        }
    }

    // теперь 2 слой
    for(let i = 1; i < NODES + 1; i++) {
        error = 0;
        for(let j = 1 + NODES; j < length - 1; j++) {
            if(links[i][j] != undefined) {
                error += activationDerivate(thirdNode(run12(x), j)) * links[i][j];
            } else {
                console.error("NO WEIGHT BUT HAS TO BE");
            }
        }
        deltas[i] = activationDerivate(secondNode(firstNode(x), i)) * error;
    }

    // обновляем веса между 3 и 4
    for(let i = 1 + NODES; i < length - 1; i++) {
        links[i][length - 1] += n * deltas[length - 1] * thirdNode(run12(x), i);
    }

    // теперь 2 и 3
    for(let i = 1; i < NODES + 1; i++) {
        for(let j = 1 + NODES; j < length - 1; j++) {
            links[i][j] += n * deltas[j] * secondNode(firstNode(x), i);
        }
    }

    // между 1 и 2
    for(let i = 1; i < NODES + 1; i++) {
        links[0][i] += + n * deltas[i] * firstNode(x);
    }
}

// ------------------------------------------------------------------------------------------------------

// функция активации
var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида

// производная функции активации
var activationDerivate = (x : number) =>
        Math.pow(Math.E, -x) / Math.pow(Math.pow(Math.E, -x) + 1, 2);

// x - число, подаваемое сети на вход. Нормализуем его.
// границы х от -10 до +10 => от 0 до 1
var firstNode = (x : number) => activation((x + 10.0) / 20.0);

// NodeNumber - номер вершины среди всех вершинок. х - выход первого слоя
function secondNode(x : number, NodeNumber : number) {
    if (getLayerOfNode(NodeNumber) !== 2) { // на всякий случай
        console.error("GOT NODE NOT OF 2 LAYER");
    }
    if (links[0][NodeNumber] == undefined) {
        console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
    }
    return activation(links[0][NodeNumber] * x);
}

// results is an array of 10 results of the second layer
// NodeNumber is still the number of the node in the full list
function thirdNode(results : number[], NodeNumber : number) {
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
    return activation(sum);
}

// an external node (final one). Returns the prediction
function externalNode(results : number[]) {
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
    return activation(sum);
}

function run12 (x : number) { // запустить 1 и 2 слой сети
    let frt = firstNode(x);
    let seconds = [];
    for(let i = 1; i < NODES + 1; i++) {
        seconds.push(secondNode(frt, i));
    }
    return seconds;
}

function run123 (x : number) { // запуск 1, 2 и 3 слоев
    let seconds = run12(x);
    let thirds = [];
    for(let i = NODES + 1; i < length - 1; i++) {
        thirds.push(thirdNode(seconds, i));
    }
    return thirds;
}

// запустить все слои (запустить сеть)
function runAll(x : number) {
    let thirds = run123(x);
    let result = externalNode(thirds);
    return result;
}

//let mistakes = []; // ошибки возбужденности
let eps = 0.1;
// let data = fs.readFileSync('data.txt', { encoding: 'utf-8' }).split('\n');
let total = 10000;
for (let i = 0; i < total; i++) {
    if (i % (total / 100)  == 0) {
        console.log("learning " + ((i * 100) / total) + "% done");
    }
    let x = Math.random() * 20 - 10;
    let result = runAll(x);
    let realValue = modeling(x);
    if (Math.abs(result - realValue) > eps) { // если сеть отвечает совсем не так, как надо, обучаем
        //console.log("function of " + x + " returned " + renormalize(result));
        //console.log("mistake is " + Math.abs(renormalize(result) - abs_modeling(x)));
        //mistakes.push(Math.abs(result - realValue));
        changeWeights(x);
    } else {
        //console.log("OK");
    }
    //console.log("-----------------------------------");
};

//console.log("finished learning. Started testing:");
total = 10000;
let successful = 0;
for (let i = 0; i < 10000; i++) {
    let x = Math.random() * 20 - 10; // random от 0 до 1 => от -10 до 10
    let result = runAll(x);
    let realValue = modeling(x);
    if (Math.abs(result - realValue) <= eps) {
        successful++;
    }
}
console.log("Testing. Procent of success is " + (100 * successful / total));

// mistakes.forEach(mistake => console.log(mistake.toString().replace('.', ',')));

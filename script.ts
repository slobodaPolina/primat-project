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
// и допустим 1 скрытый слой.
// в каждый из 10 входит х, там будет вес,
// получится 10 весов на скрытый слой
// + 10 * 10 весов на второй + 10 на финальный (внешний)

var NODES = 20; // количество вершин во внутреннем слое
var length = NODES + 2; // всего вершин. Начальная, конечная и 2 слоя между ними по NODES
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
            2 : 3;
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

    // обработаем 2 слой
    for(let i = 1; i < length - 1; i++) {
        if(links[i][length - 1] != undefined) {
            error = activationDerivate(externalNode(run12(x))) * links[i][length - 1];
            deltas[i] = activationDerivate(secondNode(firstNode(x), i)) * error;
        } else {
            console.error("NO WEIGHT BUT HAS TO BE");
        }
    }

    // обновляем веса между 2 и 3
    for(let i = 1; i < length - 1; i++) {
        links[i][length - 1] += n * deltas[length - 1] * secondNode(firstNode(x), i);
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

// an external node (final one). Returns the prediction
function externalNode(results : number[]) {
    var sum = 0;
    results.forEach(
        (result, index) => {
            if (links[1 + index][length - 1] == undefined) {
                console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
            }
            // аналогично перебираем 2 слой,
            // домножаем на веса из вершины 2 слоя в последнюю (вершину 3 слоя)
            sum += result * links[1 + index][length - 1];
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

// запустить все слои (запустить сеть)
function runAll(x : number) {
    let seconds = run12(x);
    let result = externalNode(seconds);
    return result;
}

//let mistakes = []; // ошибки возбужденности
let eps = 0.1;
// let data = fs.readFileSync('data.txt', { encoding: 'utf-8' }).split('\n');

for (let i = 0; i < 10; i++) {
    let x = 0;
    let result = runAll(x);
    let realValue = modeling(x);
    if (Math.abs(result - realValue) > eps) { // если сеть отвечает совсем не так, как надо, обучаем
        console.log("function of " + x + " returned " + renormalize(result));
        console.log("mistake is " + Math.abs(renormalize(result) - abs_modeling(x)));
        //mistakes.push(Math.abs(result - realValue));
        changeWeights(x);
    } else {
        console.log("OK");
    }
    //console.log("-----------------------------------");
};

/*let total = 50000;
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
*/

// mistakes.forEach(mistake => console.log(mistake.toString().replace('.', ',')));

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

var NODES = 10; // количество вершин во внутреннем слое
var length = NODES + 3; // всего вершин. Начальная, смещение, конечная и NODES
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
    return NodeNumber < 2 ?
        1 :
        NodeNumber > 1 && NodeNumber < length - 1 ? 2 : 3;
}

// ------------------------------------------------------------------------------------------------------------

// меняет веса и n на каждом этапе обучения
function changeWeights (x : number, outputs : number[]) {
    let deltas = Array(length);
    let result = outputs[length - 1];
    let error = result - modeling(x);
    deltas[length - 1] = activationDerivate(result) * error; // насколько изменить выходной результат

    // обработаем 2 слой. Распространяем ошибку
    for(let i = 2; i < length - 1; i++) {
        if(links[i][length - 1] != undefined) {
            links[i][length - 1] -= outputs[i] * deltas[length - 1] * n;
            error = links[i][length - 1] * deltas[length - 1];
            deltas[i] = activationDerivate(outputs[i]) * error;
        } else {
            console.error("NO WEIGHT BUT HAS TO BE");
        }
    }

    // 1 слой
    for(let i = 2; i < length - 1; i++) {
        links[0][i] -= outputs[0] * deltas[i];
        links[1][i] -= outputs[1] * deltas[i];
    }
}

// ------------------------------------------------------------------------------------------------------

// функция активации
var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида

// производная функции активации
var activationDerivate = (x : number) => activation(x) * (1 - activation(x));

// x - число, подаваемое сети на вход. Нормализуем его.
// границы х от -10 до +10 => от 0 до 1
var firstNode = (x : number) => activation((x + 10.0) / 20.0);

// NodeNumber - номер вершины среди всех вершинок. х - выход первого слоя
function secondNode(x : number[], NodeNumber : number) {
    if (getLayerOfNode(NodeNumber) !== 2) { // на всякий случай
        console.error("GOT NODE NOT OF 2 LAYER");
        console.log(NodeNumber);
    }
    if (links[0][NodeNumber] == undefined || links[1][NodeNumber] == undefined) {
        console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
    }
    return activation(links[0][NodeNumber] * x[0] + links[1][NodeNumber] * x[1]);
}

// an external node (final one). Returns the prediction
function externalNode(outputs : number[]) {
    var sum = 0;
    outputs.forEach(
        (output, index) => {
            if (getLayerOfNode(index) != 2) { // выбираем только 2 слой
                return;
            }
            if (links[index][length - 1] == undefined) {
                console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
            }
            // аналогично перебираем 2 слой,
            // домножаем на веса из вершины 2 слоя в последнюю (вершину 3 слоя)
            sum += output * links[index][length - 1];
        }
    );
    outputs[length - 1] = activation(sum);
    return outputs;
}

function run12 (x : number, outputs : number[]) { // запустить 1 и 2 слой сети
    outputs[0] = firstNode(x);
    outputs[1] = activation(1);
    for(let i = 2; i < length - 1; i++) {
        outputs[i] = secondNode([outputs[0], outputs[1]], i);
    }
    return outputs;
}

// запустить все слои (запустить сеть)
function runAll(x : number) {
    let outputs = Array(length);
    outputs = run12(x, outputs);
    return externalNode(outputs);
}

function square(real : number, expecting : number) {
    return Math.pow(real - expecting, 2);
}

// получить среднее квадратическое отклонение по массивам
function arrSquare(real : number[], expecting : number[]) {
    if(real.length != expecting.length) {
        console.error("LENGTHS ARE DIFFERENT!");
    } else {
        let sum = 0;
        real.forEach((el, index) => {
            sum += square(el, expecting[index]);
        });
        return sum/real.length;
    }
}

function printForExcel(el : number) {
    console.log(el.toString().replace(".", ","));
}
// ---------------------------------------------------------------------------------------------

var n = 0.5; // скорость обучения, нужно подбирать
var goal = 0.01; // на каком среднем значениии квадрата разности можно закончить (в абсолютных величинах)
let LEARNING = false;

if (LEARNING) {
    let data = fs.readFileSync('data.txt', { encoding: 'utf-8' }).split('\n');
    let training_loss = [];
    let previous_sq : number;

    do {
        data.forEach(s => {
            let x = Number(s); // для каждой точки из файла
            let outputs = runAll(x);
            changeWeights(x, outputs);
        });
        // теперь для поправленных весов смотрим на вывод
        let predictions = [];
        let correct = [];
        data.forEach(s => {
            let x = Number(s);
            predictions.push(renormalize(runAll(x)[length - 1]));
            correct.push(abs_modeling(x));
        });
        previous_sq = arrSquare(predictions, correct);
        training_loss.push(previous_sq);
        printForExcel(previous_sq);
    } while (previous_sq > goal);

    var file = fs.createWriteStream('weights.txt');
    for(let i = 0; i < length; i++) {
        for(let j = 0; j < length; j++) {
            file.write(links[i][j] + ' ');
        }
        file.write('\n');
    }
    file.end();
    console.log("FINISHED LEARNING!");
} else {
    var text = fs.readFileSync('weights.txt', "utf-8");
    let lines = text.split('\n');
    let strings = Array(length);
    for (let i = 0; i < length; i++) {
        strings[i] = lines[i].split(" ");
    }
    for(let i = 0; i < length; i++) {
        for(let j = 0; j < length; j++){
            links[i][j] = strings[i][j] * 1;
            if (isNaN(links[i][j])) {
                links[i][j] = undefined;
            }
        }
    }
    let successful = 0;
    let total = 10000;
    let eps = 0.01; // с какой точностью достаточно получать ответ (в величинах возбужденности)
    for (let i = 0; i < total; i++) {
        let x = Math.random() * 20 - 10; // random от 0 до 1 => от -10 до 10
        let result = runAll(x)[length - 1];
        let realValue = modeling(x);
        if (Math.abs(result - realValue) <= eps) {
            successful++;
        }
    }
    console.log("Testing. Procent of success is " + (100 * successful / total));
}

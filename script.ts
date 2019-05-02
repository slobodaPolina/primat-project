import * as fs from 'fs';

function modeling(x : number, y : number, z : number) {
    var text = fs.readFileSync('data.txt', "utf-8");
    let lines = text.split('\n');
    let line = lines.find(line => line.startsWith(x + " " + y + " " + z));
    return parseInt(line.charAt(line.length - 1));
 }

// выходная вершина дает ответ
// и допустим 1 скрытый слой.
// в каждый из 10 входит х, там будет вес,
// получится 10 весов на скрытый слой
// + 10 * 10 весов на второй + 10 на финальный (внешний)

var NODES = 10; // количество вершин во внутреннем слое
var length = NODES + 5; // всего вершин. Начальные 3, смещение, конечная и NODES
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
    return NodeNumber < 4 ?
        1 :
        NodeNumber > 3 && NodeNumber < length - 1 ? 2 : 3;
}

// ------------------------------------------------------------------------------------------------------------

// меняет веса и n на каждом этапе обучения
function changeWeights (x : number, y : number, z : number, outputs : number[]) {
    let deltas = Array(length);
    let result = outputs[length - 1];
    let error = result - modeling(x, y, z);
    deltas[length - 1] = activationDerivate(result) * error; // насколько изменить выходной результат

    // обработаем 2 слой. Распространяем ошибку
    for(let i = 4; i < length - 1; i++) {
        if(links[i][length - 1] != undefined) {
            links[i][length - 1] -= outputs[i] * deltas[length - 1] * n;
            error = links[i][length - 1] * deltas[length - 1];
            deltas[i] = activationDerivate(outputs[i]) * error;
        } else {
            console.error("NO WEIGHT BUT HAS TO BE");
        }
    }

    // 1 слой
    for(let i = 4; i < length - 1; i++) {
        for(let k = 0; k < 4; k++) {
            links[k][i] -= outputs[k] * deltas[i];
        }
    }
}

// ------------------------------------------------------------------------------------------------------

// функция активации
var activation = (x : number) => 1 / (1 + Math.pow(Math.E, -x)); // сигмоида

// производная функции активации
var activationDerivate = (x : number) => activation(x) * (1 - activation(x));

// NodeNumber - номер вершины среди всех вершинок. х - выход первого слоя
function secondNode(x : number[], NodeNumber : number) {
    if (getLayerOfNode(NodeNumber) !== 2) { // на всякий случай
        console.error("GOT NODE NOT OF 2 LAYER");
        console.log(NodeNumber);
    }
    if (links[0][NodeNumber] == undefined ||
        links[1][NodeNumber] == undefined ||
        links[2][NodeNumber] == undefined ||
        links[3][NodeNumber] == undefined
    ) {
        console.error("WEIGHT IS UNDEFINED BUT SHOULDNOT");
    }
    return activation(links[0][NodeNumber] * x[0] +
        links[1][NodeNumber] * x[1] +
        links[2][NodeNumber] * x[2] +
        links[3][NodeNumber] * x[3]
    );
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

function run12 (x : number, y : number, z : number, outputs : number[]) { // запустить 1 и 2 слой сети
    outputs[0] = activation(x);
    outputs[1] = activation(y);
    outputs[2] = activation(z);
    outputs[3] = activation(1);
    for(let i = 4; i < length - 1; i++) {
        outputs[i] = secondNode([outputs[0], outputs[1], outputs[2], outputs[3]], i);
    }
    return outputs;
}

// запустить все слои (запустить сеть)
function runAll(x : number, y : number, z : number) {
    let outputs = Array(length);
    outputs = run12(x, y, z, outputs);
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
    let training_loss = [];
    let previous_sq : number;

    do {
        for(let i = 0; i < 2; i++) {
            for(let j = 0; j < 2; j++) {
                for(let k = 0; k < 2; k++) {
                    let outputs = runAll(i, j, k);
                    changeWeights(i, j, k, outputs);
                }
            }
        }

        // теперь для поправленных весов смотрим на вывод
        let predictions = [];
        let correct = [];
        for(let i = 0; i < 2; i++) {
            for(let j = 0; j < 2; j++) {
                for(let k = 0; k < 2; k++) {
                    predictions.push(runAll(i, j, k)[length - 1]);
                    correct.push(modeling(i, j, k));
                }
            }
        };
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
    for(let i = 0; i < 2; i++) {
        for(let j = 0; j < 2; j++) {
            for(let k = 0; k < 2; k++) {
                let result = runAll(i, j, k)[length - 1];
                let realValue = modeling(i, j, k);
                if (Math.abs(result - realValue) < 0.5) {
                    successful++;
                }
            }
        }
    }
    console.log("Testing. Got " + successful + "/" + 8);
}

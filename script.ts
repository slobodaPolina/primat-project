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
        NodeNumber > 0 && NodeNumber <= NODES ? 2 : 3;
}

// ------------------------------------------------------------------------------------------------------------

// меняет веса и n на каждом этапе обучения
function changeWeights (x : number, outputs : number[]) {
    let deltas = Array(length);
    let result = outputs[length - 1];
    let error = result - modeling(x);
    deltas[length - 1] = activationDerivate(result) * error; // насколько изменить выходной результат
    //console.log(result);
    //console.log(activationDerivate(result));
    //console.log(deltas[length - 1]);

    // обработаем 2 слой. Распространяем ошибку
    for(let i = 1; i < length - 1; i++) {
        if(links[i][length - 1] != undefined) {
            links[i][length - 1] -= outputs[i] * deltas[length - 1] * n;
            error = links[i][length - 1] * deltas[length - 1];
            deltas[i] = activationDerivate(outputs[i]) * error;
        } else {
            console.error("NO WEIGHT BUT HAS TO BE");
        }
    }

    // 1 слой
    for(let i = 1; i < length - 1; i++) {
        links[0][i] -= outputs[0] * deltas[i];
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
    for(let i = 1; i < NODES + 1; i++) {
        outputs[i] = secondNode(outputs[0], i);
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
// ---------------------------------------------------------------------------------------------

let eps = 0.1;
let epohs = 100;
let data = fs.readFileSync('data.txt', { encoding: 'utf-8' }).split('\n');
let training_loss = [];

for (let i = 0; i < epohs; i++) {
    data.forEach(s => {
        let x = Number(s); // для каждой точки из файла запускаем и корректируем, если нужно
        let outputs = runAll(x);
        let result = outputs[length - 1];
        let realValue = modeling(x);
        if (Math.abs(result - realValue) > eps) { // если сеть отвечает совсем не так, как надо, обучаем
            changeWeights(x, outputs);
        }
    });
    // теперь для поправленных весов смотрим на вывод
    let predictions = [];
    let correct = [];
    data.forEach(s => {
        let x = Number(s);
        predictions.push(runAll(x)[length - 1]);
        correct.push(modeling(x));
    });
    training_loss.push(arrSquare(predictions, correct));
};

console.log(training_loss);

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

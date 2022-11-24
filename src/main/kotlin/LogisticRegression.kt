import space.kscience.plotly.*
import space.kscience.plotly.models.ScatterMode
import kotlin.math.exp
import kotlin.math.roundToInt
import kotlin.random.Random

fun Double.roundTo2(): Double = (this * 100).roundToInt().toDouble() / 100

fun generateDataset(size: Int): List<Pair<Double, Double>> {
    val firstPart = (1..size).map {
        Pair(Random.nextDouble(START, BORDER_LEFT).roundTo2(), 0.0)
    }
    val secondPart = (1..size).map {
        Pair(Random.nextDouble(BORDER_RIGHT, END).roundTo2(), 1.0)
    }
    return firstPart + secondPart
}

var w = 0.0
var b = 0.0
const val START = 1.0
const val BORDER_LEFT = 3.3
const val BORDER_RIGHT = 3.2
const val END = 5.0
const val DATASET_SIZE = 100

fun sigmoid(x: Double) = 1 / (1 + exp(-x))

fun logit(x: Double) = w * x + b

fun predict(x: Double) = sigmoid(logit(x))

fun train(data: List<Pair<Double, Double>>, epochs: Int, lr: Double) {
    for (i in 0..epochs) {
        var dw = 0.0
        var db = 0.0
        for (point in data) {
            dw += lr * (point.second - predict(point.first))
            db += lr * point.first * (point.second - predict(point.first))
        }
        w += db; b += dw
    }
}

fun main() {
    val data = generateDataset(DATASET_SIZE)
    train(data, 10000, 0.05)
    val dataX = data.map { it.first }
    val dataY = data.map { it.second }

    val resX = ((START * 100).toInt()..(END * 100).toInt()).map { it * 0.01 }
    val resY = resX.map { predict(it) }

    val plot = Plotly.plot {
        scatter {
            x.set(dataX)
            y.set(dataY)
            mode = ScatterMode.markers
            name = "Исходные данные"
        }

        trace {
            x.set(resX)
            y.set(resY)
            name = "Кривая регрессии"
        }

        layout {
            title = "Логистическая регрессия"
        }
    }
    print("$w $b")
    plot.makeFile()
}
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import java.io.File
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

class BayesianClassification {

    private var ev0 = DoubleArray(5)
    private var ev1 = DoubleArray(5)
    private var sd0 = DoubleArray(5)
    private var sd1 = DoubleArray(5)
    private var c0 = 0.0
    private var c1 = 0.0
    private var p0 = 0.0
    private var p1 = 0.0

    private fun gaussian(x: Double, m: Double, sd: Double): Double =
        (1 / (sd * sqrt(2 * 3.14))) * exp((-(x - m).pow(2)) / (2 * sd.pow(2)))

    fun prepare(data: List<List<Double>>) {
        for (row in data) {
            if (row[5] == 0.0) {
                for (i in (0..4))
                    ev0[i] += row[i]
                c0++
            } else {
                for (i in (0..4))
                    ev1[i] += row[i]
                c1++
            }
        }
        ev0 = ev0.map { it / c0 }.toDoubleArray()
        ev1 = ev1.map { it / c1 }.toDoubleArray()
        p0 = c0 / (c0 + c1)
        p1 = c1 / (c0 + c1)

        for (row in data) {
            if (row[5] == 0.0) {
                for (i in (0..4))
                    sd0[i] += (row[i] - ev0[i]).pow(2)
            } else {
                for (i in (0..4))
                    sd1[i] += (row[i] - ev1[i]).pow(2)
            }
        }
        sd0 = sd0.map { sqrt(it / c0) }.toDoubleArray()
        sd1 = sd1.map { sqrt(it / c1) }.toDoubleArray()
    }

    fun predict(x: DoubleArray): Boolean {
        val ps0 = DoubleArray(5)
        val ps1 = DoubleArray(5)
        for (i in (0..4)) {
            ps0[i] = gaussian(x[i], ev0[i], sd0[i])
            ps1[i] = gaussian(x[i], ev1[i], sd1[i])
        }

        val finalP0 = p0 * ps0.reduce { acc, i -> acc * i }
        val finalP1 = p1 * ps1.reduce { acc, i -> acc * i }
        return finalP0 < finalP1
    }
}

fun doubleToBoolean(x: Double) = x == 1.0

fun main() {
    val trainMultiplier = 0.75

    val file = File("dataset.csv")
    val tableRaw: List<List<String>> = csvReader().readAll(file).drop(1)
    val tableTrain: MutableList<List<Double>> = mutableListOf()
    val tableVerify: MutableList<List<Double>> = mutableListOf()

    for (row in tableRaw.subList(0, (trainMultiplier * tableRaw.size.toDouble()).toInt()))
        tableTrain.add(row.map { it.toDouble() })

    for (row in tableRaw.subList((trainMultiplier * tableRaw.size.toDouble()).toInt(), tableRaw.size))
        tableVerify.add(row.map { it.toDouble() })

    val bc = BayesianClassification()
    bc.prepare(tableTrain)

    var acc = 0
    for (row in tableVerify) {
        val predicted = bc.predict(row.subList(0,5).toDoubleArray())
        val isCorrect = predicted == doubleToBoolean(row[5])
        if (isCorrect) acc++
    }
    val result = acc.toDouble() / tableVerify.size
    println(acc)
    println(tableVerify.size)
    println("Точность алгоритма: ${result.roundTo2()}")
}

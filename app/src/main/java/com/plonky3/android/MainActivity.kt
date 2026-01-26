package com.plonky3.android

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    companion object {
        init {
            System.loadLibrary("plonky3_jni")
        }
    }

    private external fun runPoseidonTest(input: IntArray): IntArray

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val status = findViewById<TextView>(R.id.status)
        val input = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8)

        val output = try {
            runPoseidonTest(input)
        } catch (e: UnsatisfiedLinkError) {
            status.text = "Native library not loaded: ${e.message}"
            return
        }

        status.text = "Input: ${input.joinToString()}\nOutput: ${output.joinToString()}"
    }
}

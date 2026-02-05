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

    private external fun runFibAirZk(): String
    private external fun setBackend(backend: String)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val status = findViewById<TextView>(R.id.status)

        val output = try {
            setBackend("vulkan")
            runFibAirZk()
        } catch (e: UnsatisfiedLinkError) {
            status.text = "Native library not loaded: ${e.message}"
            return
        }

        status.text = output
    }
}

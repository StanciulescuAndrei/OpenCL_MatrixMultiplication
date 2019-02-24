using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL;
using System.IO;
using System.Diagnostics;

namespace OpenCL_MatrixMultiplication
{
    class Program
    {
        static void Main(string[] args)
        {


            OpenCL.OpenCL cl = new OpenCL.OpenCL();
            cl.Accelerator = AcceleratorDevice.GPU;

            string kernel = File.ReadAllText(@"../../../kernel.cl");

            const int aX = 2000, aY = 2000, bX = 2000, bY = 2000; //x=number of lines, y = number of columns
            double[] a = new double[aX * aY];
            double[] b = new double[bX * bY];
            double[] c = new double[aX * bY]; // resulting matrix, with aX*bY dimensions
            int[] dimensions = new int[3] { aX, aY, bY };

            Random rand = new Random();
            for(int i = 0; i < aX; i++)
            {
                for(int j=0;j<aY;j++)
                {
                    a[i*aY + j] = rand.NextDouble();
                }
            }
            for (int i = 0; i < bX; i++)
            {
                for (int j = 0; j < bY; j++)
                {
                    b[i * bY + j] = rand.NextDouble();
                }
            }

            cl.SetKernel(kernel, "MatrixMulti");
            //Pass the dimensions and matrix pointers:
            cl.SetParameter(dimensions, a, b, c);
            //Each cell of the result will be computed at the same time
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            cl.Execute(aX * bY);
            stopwatch.Stop();

            Console.WriteLine("GPU parallel multiplication done in " + stopwatch.ElapsedMilliseconds/1000.00 + " seconds");
            for (int i = 0; i < aX; i++)
            {
                for (int j = 0; j < bY; j++)
                {
                    //Console.Write(c[i*bY + j] + " ");
                }
                //Console.WriteLine();
            }
            

            //Run linearily on CPU:
            stopwatch.Reset();
            stopwatch.Start();
            for(int id = 0;id<aX * bY; id++)
            {
                int NLin_1 = dimensions[0]; //number of lines of first matrix
                int NCol_1 = dimensions[1]; //number of columns of first matrix
                int NCol_2 = dimensions[2]; //number of columns of second matrix

                int L = id / NCol_2; //get the position in the final matrix from the id
                int C = id - L * NCol_2;

                double element = 0;
                for (int i = 0; i < NCol_1; i++)
                {
                    element = element + a[L * NCol_1 + i] * b[C + NCol_2 * i];
                }
                c[id] = element;
            }
            stopwatch.Stop();
            Console.WriteLine("CPU linear multiplication done in " + stopwatch.ElapsedMilliseconds / 1000.00 + " seconds");
            stopwatch.Reset();
            stopwatch.Start();
            Parallel.For(0, aX * bY, id =>
            {
                int NLin_1 = dimensions[0]; //number of lines of first matrix
                int NCol_1 = dimensions[1]; //number of columns of first matrix
                int NCol_2 = dimensions[2]; //number of columns of second matrix

                int L = id / NCol_2; //get the position in the final matrix from the id
                int C = id - L * NCol_2;

                double element = 0;
                for (int i = 0; i < NCol_1; i++)
                {
                    element = element + a[L * NCol_1 + i] * b[C + NCol_2 * i];
                }
                c[id] = element;
            }
            );
            
            stopwatch.Stop();
            Console.WriteLine("CPU parallel multiplication done in " + stopwatch.ElapsedMilliseconds / 1000.00 + " seconds");

            Console.ReadKey();
        }
    }
}

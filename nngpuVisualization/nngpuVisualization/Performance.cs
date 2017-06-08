using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace nngpuVisualization
{
    class Performance
    {
        private Stopwatch _timer = new Stopwatch();

        public void Start()
        {
            _timer.Start();
        }

        public long Stop()
        {
            _timer.Stop();
            return _timer.ElapsedMilliseconds;
        }
    }
}

s = 100000
x = tf.Variable(tf.random_normal([s,s]))
y = tf.Variable(tf.random_normal([s,s]))
f = tf.matmul(x, y)
calculations = 2*s*s*s
init = tf.global_variables_initializer()

with tf.Session() as sess:
   init.run()
   start = time.time()
   f.eval()
   end = time.time()
   elapsed = end - start
   flops = (calculations/elapsed)/1e12
   print("%d by %d matricies: %.5f TFLOPS" % (s, s, flops))
sess.close(
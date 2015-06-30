
(function() {
	var LOG_2PI = 1.8378770664093453;
	function gaussianScore(x, mu, sigma) {
		return -0.5 * (LOG_2PI + 2 * Math.log(sigma) + (x - mu) * (x - mu) / (sigma * sigma));
	}

	function makeRunTest(N) {
		return function(mu, sigma) {
			var xs = [];
			for (var i = 0; i < N; i++) xs.push(Math.random());
			var sum = 0;
			for (var i = 0; i < N; i++) {
				var x = xs[i];
				if (Math.random() > 0.5)
					sum = sum + gaussianScore(x, mu, sigma);
				else
					sum = sum - gaussianScore(x, mu, sigma) + 1.0;
			}
			return sum;
		}
	}

	return makeRunTest;
});

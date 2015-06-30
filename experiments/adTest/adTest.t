
local ad = require('ad');

local random = terralib.includecstring [[
#include <stdlib.h>
double __random() { return rand() / (double)(RAND_MAX); }
]].__random;


-- The test model, parameterized by numeric type T
local function genModel(T)
	return function(N)
		local LOG_2PI = 1.8378770664093453
		local terra gaussianScore(x: T, mu: T, sigma: T)
			return -0.5 * (LOG_2PI + 2 * ad.math.log(sigma) + (x - mu) * (x - mu) / (sigma * sigma))
		end
		return terra(mu: T, sigma: T)
			var xs: T[N]
			for i=0,N do
				xs[i] = random()
			end
			var sum = T(0.0)
			for i=0,N do
				var x = xs[i]
				if (random() > 0.5) then
					sum = sum + gaussianScore(x, mu, sigma)
				else
					sum = sum - gaussianScore(x, mu, sigma) + 1.0
				end
			end
			return sum
		end
	end	
end

-- Set up different versions of the code: ADed and non-ADed
local makeRunTest = genModel(double)
local makeRunTestAD = genModel(ad.num)

-- Run tests at various data sizes and conditions
local conditions = {
	'noad',
	-- no adPrimal condition, because we have perfect static operator overloading
	'adDual',
	'adGradient'
}
local sizes = {}
for i=100,1000,100 do table.insert(sizes, i) end
local numRuns = 100
local mu = 0.5
local sigma = 1.5
local csvFile = io.open('./experiments/adTest/adTest_terra.csv', 'w')
csvFile:write('condition,size,time\n')
for _,condition in ipairs(conditions) do
	print('CONDITION = ' .. condition)
	local makeTest = condition == 'noad' and makeRunTest or makeRunTestAD
	for _,size in ipairs(sizes) do
		print('  size = ' .. size)
		local f = makeTest(size)
		local testfn = terra(mu: double, sigma: double)
			var res = f(mu, sigma)
			escape
				if condition == 'adGradient' then
					emit quote res:grad() end
				elseif condition == 'adDual' then
					emit quote ad.recoverMemory() end
				end
			end
		end
		-- Actually run
		for r=1,numRuns do
			local t0 = terralib.currenttimeinseconds()
			testfn(mu, sigma)
			local t1 = terralib.currenttimeinseconds()
			local tdiff = t1 - t0
			csvFile:write(string.format('%s,%u,%g\n', condition, size, tdiff))
		end
	end
end
csvFile:close()




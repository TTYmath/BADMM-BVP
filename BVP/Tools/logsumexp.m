function sa = logsumexp(a,dim)
ma = max(a,[],dim);
sa = ma+log(sum(exp(a-ma),dim));
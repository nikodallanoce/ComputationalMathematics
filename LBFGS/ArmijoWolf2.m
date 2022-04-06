function a = ArmijoWolf2(xk,pk,f,f1,c1,c2,ai,amax)
i=1;
max_iter = 1000;
aold = 0;
while i < max_iter
    alpha_f = @(alpha)xk + alpha.*pk; % alphafunction lambda alpha: xk+aplha*pk
    phia = f(alpha_f(ai));
    phiz = f(xk);
    if (phia > phiz+c1*ai*f1(xk) || (i>1 && phia >= aold))
        a=zoom(aold,ai);
        break;
    end
    phia1 = f1(phia);
    if abs(phia1) <= -c2*f1(xk)
        a=ai;
    end
    if phia1>=0
        a=zoom(ai,amax);
        break;
    end
    aold = ai;
    ai = (ai*tau)+ai; %choose ai+1 in (ai,amax)
    i=i+1;
end

function aitar = zoom(alo,ahi)
    while true
      aj = (ahi-alo)/2; %interpolation step to find aj in (ahi, alo).
      phiaj = f(alpha_f(aj));
      phialo = f(alpha_f(alo));
      if (phiaj >= phiz+c1*aj*f1(xk) || (phiaj>=phialo))
          ahi = aj;
      else
          phi1aj = f1(alpha_f(aj));
          if abs(phi1aj)<= -c2*f1(xk)
              aitar = aj;
              break;
          end
          if phi1aj*(ahi-alo)>=0
              ahi=aho;
          end
          alo=aj;
      end
    end
end

end


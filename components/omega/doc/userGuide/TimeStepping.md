(omega-user-time-stepping)=

# Time stepping
Time stepper referes to the numerical scheme used to advance the model in time.
Omega implements a couple of time stepping options. Thee can be chosen in the
config file. The defualts is the forward-backward scheme.

```yaml
    TimeStepping:
       TimeStepperType: 'Forward-Backward'
```

The following time steppers are currently available:
| Name | Description |
| ------------------- | ------- |
| Forward-Backward | kinetic energy of horizontal velocity on cells
| RungeKutta2 | kinetic energy of horizontal velocity on cells
| RungeKutta4 | divergence of horizontal velocity

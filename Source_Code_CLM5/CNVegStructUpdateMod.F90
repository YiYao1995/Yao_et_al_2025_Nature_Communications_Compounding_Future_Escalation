module CNVegStructUpdateMod

  !-----------------------------------------------------------------------
  ! Module for vegetation structure updates (LAI, SAI, htop, hbot)
  !
  ! !USES:
  use shr_kind_mod         , only: r8 => shr_kind_r8
  use shr_const_mod        , only : SHR_CONST_PI
  use clm_varctl           , only : iulog, use_cndv
  use CNDVType             , only : dgv_ecophyscon    
  use WaterDiagnosticBulkType       , only : waterdiagnosticbulk_type
  use FrictionVelocityMod  , only : frictionvel_type
  use CNDVType             , only : dgvs_type
  use CNVegStateType       , only : cnveg_state_type
  use CropType             , only : crop_type
  use CNVegCarbonStateType , only : cnveg_carbonstate_type
  use CanopyStateType      , only : canopystate_type
  use PatchType            , only : patch                
  !
  implicit none
  private
  !
  ! !PUBLIC MEMBER FUNCTIONS:
  public :: CNVegStructUpdate
  !-----------------------------------------------------------------------

contains

  !-----------------------------------------------------------------------
  subroutine CNVegStructUpdate(num_soilp, filter_soilp, &
       waterdiagnosticbulk_inst, frictionvel_inst, dgvs_inst, cnveg_state_inst, crop_inst, &
       cnveg_carbonstate_inst, canopystate_inst)
    !
    ! !DESCRIPTION:
    ! On the radiation time step, use C state variables and epc to diagnose
    ! vegetation structure (LAI, SAI, height)
    !
    ! !USES:
    use pftconMod        , only : noveg, nc3crop, nc3irrig_dri, nc3irrig_spr, nc3irrig_fld, nbrdlf_evr_shrub, nbrdlf_dcd_brl_shrub
    use pftconMod        , only : npcropmin 
    use pftconMod        , only : ntmp_corn, nirrig_dri_tmp_corn, nirrig_spr_tmp_corn, nirrig_fld_tmp_corn
    use pftconMod        , only : ntrp_corn, nirrig_dri_trp_corn, nirrig_spr_trp_corn, nirrig_fld_trp_corn
    use pftconMod        , only : nsugarcane, nirrig_dri_sugarcane, nirrig_spr_sugarcane, nirrig_fld_sugarcane
    use pftconMod        , only : nmiscanthus, nirrig_dri_miscanthus, nirrig_spr_miscanthus, nirrig_fld_miscanthus, nswitchgrass, nirrig_dri_switchgrass, nirrig_spr_switchgrass, nirrig_fld_switchgrass
    
    use pftconMod        , only : pftcon
    use clm_varctl       , only : spinup_state
    use clm_time_manager , only : get_rad_step_size
    !
    ! !ARGUMENTS:
    integer                      , intent(in)    :: num_soilp       ! number of column soil points in patch filter
    integer                      , intent(in)    :: filter_soilp(:) ! patch filter for soil points
    type(waterdiagnosticbulk_type)        , intent(in)    :: waterdiagnosticbulk_inst
    type(frictionvel_type)       , intent(in)    :: frictionvel_inst
    type(dgvs_type)              , intent(in)    :: dgvs_inst
    type(cnveg_state_type)       , intent(inout) :: cnveg_state_inst
    type(crop_type)              , intent(in)    :: crop_inst
    type(cnveg_carbonstate_type) , intent(in)    :: cnveg_carbonstate_inst
    type(canopystate_type)       , intent(inout) :: canopystate_inst
    !
    ! !REVISION HISTORY:
    ! 10/28/03: Created by Peter Thornton
    ! 2/29/08, David Lawrence: revised snow burial fraction for short vegetation
    !
    ! !LOCAL VARIABLES:
    integer  :: p,c,g      ! indices
    integer  :: fp         ! lake filter indices
    real(r8) :: taper      ! ratio of height:radius_breast_height (tree allometry)
    real(r8) :: stocking   ! #stems / ha (stocking density)
    real(r8) :: ol         ! thickness of canopy layer covered by snow (m)
    real(r8) :: fb         ! fraction of canopy layer covered by snow
    real(r8) :: tlai_old   ! for use in Zeng tsai formula
    real(r8) :: tsai_old   ! for use in Zeng tsai formula
    real(r8) :: tsai_min   ! PATCH derived minimum tsai
    real(r8) :: tsai_alpha ! monthly decay rate of tsai
    real(r8) :: dt         ! radiation time step (sec)

    real(r8), parameter :: dtsmonth = 2592000._r8 ! number of seconds in a 30 day month (60x60x24x30)
    !-----------------------------------------------------------------------
    ! tsai formula from Zeng et. al. 2002, Journal of Climate, p1835
    !
    ! tsai(p) = max( tsai_alpha(ivt(p))*tsai_old + max(tlai_old-tlai(p),0_r8), tsai_min(ivt(p)) )
    ! notes:
    ! * RHS tsai & tlai are from previous timestep
    ! * should create tsai_alpha(ivt(p)) & tsai_min(ivt(p)) in pftconMod.F90 - slevis
    ! * all non-crop patches use same values:
    !   crop    tsai_alpha,tsai_min = 0.0,0.1
    !   noncrop tsai_alpha,tsai_min = 0.5,1.0  (includes bare soil and urban)
    !-------------------------------------------------------------------------------
    
    associate(                                                            & 
         ivt                =>  patch%itype                               , & ! Input:  [integer  (:) ] patch vegetation type                                

         woody              =>  pftcon%woody                            , & ! Input:  binary flag for woody lifeform (1=woody, 0=not woody)
         slatop             =>  pftcon%slatop                           , & ! Input:  specific leaf area at top of canopy, projected area basis [m^2/gC]
         dsladlai           =>  pftcon%dsladlai                         , & ! Input:  dSLA/dLAI, projected area basis [m^2/gC]           
         z0mr               =>  pftcon%z0mr                             , & ! Input:  ratio of momentum roughness length to canopy top height (-)
         displar            =>  pftcon%displar                          , & ! Input:  ratio of displacement height to canopy top height (-)
         dwood              =>  pftcon%dwood                            , & ! Input:  density of wood (gC/m^3)                          
         ztopmx             =>  pftcon%ztopmx                           , & ! Input:
         laimx              =>  pftcon%laimx                            , & ! Input:
         
         allom2             =>  dgv_ecophyscon%allom2                   , & ! Input:  [real(r8) (:) ] ecophys const                                     
         allom3             =>  dgv_ecophyscon%allom3                   , & ! Input:  [real(r8) (:) ] ecophys const                                     

         nind               =>  dgvs_inst%nind_patch                    , & ! Input:  [real(r8) (:) ] number of individuals (#/m**2)                    
         fpcgrid            =>  dgvs_inst%fpcgrid_patch                 , & ! Input:  [real(r8) (:) ] fractional area of patch (pft area/nat veg area)    

         snow_depth         =>  waterdiagnosticbulk_inst%snow_depth_col          , & ! Input:  [real(r8) (:) ] snow height (m)                                   

         forc_hgt_u_patch   =>  frictionvel_inst%forc_hgt_u_patch       , & ! Input:  [real(r8) (:) ] observational height of wind at patch-level [m]     

         leafc              =>  cnveg_carbonstate_inst%leafc_patch      , & ! Input:  [real(r8) (:) ] (gC/m2) leaf C                                    
         deadstemc          =>  cnveg_carbonstate_inst%deadstemc_patch  , & ! Input:  [real(r8) (:) ] (gC/m2) dead stem C                               

         farea_burned       =>  cnveg_state_inst%farea_burned_col       , & ! Input:  [real(r8) (:) ] F. Li and S. Levis                                 
         htmx               =>  cnveg_state_inst%htmx_patch             , & ! Output: [real(r8) (:) ] max hgt attained by a crop during yr (m)          
         peaklai            =>  cnveg_state_inst%peaklai_patch          , & ! Output: [integer  (:) ] 1: max allowed lai; 0: not at max                  

         harvdate           =>  crop_inst%harvdate_patch                , & ! Input:  [integer  (:) ] harvest date                                       

         ! *** Key Output from CN***
         tlai               =>  canopystate_inst%tlai_patch             , & ! Output: [real(r8) (:) ] one-sided leaf area index, no burying by snow      
         tsai               =>  canopystate_inst%tsai_patch             , & ! Output: [real(r8) (:) ] one-sided stem area index, no burying by snow      
         htop               =>  canopystate_inst%htop_patch             , & ! Output: [real(r8) (:) ] canopy top (m)                                     
         hbot               =>  canopystate_inst%hbot_patch             , & ! Output: [real(r8) (:) ] canopy bottom (m)                                  
         elai               =>  canopystate_inst%elai_patch             , & ! Output: [real(r8) (:) ] one-sided leaf area index with burying by snow    
         esai               =>  canopystate_inst%esai_patch             , & ! Output: [real(r8) (:) ] one-sided stem area index with burying by snow    
         frac_veg_nosno_alb =>  canopystate_inst%frac_veg_nosno_alb_patch & ! Output: [integer  (:) ] frac of vegetation not covered by snow [-]         
         )

      dt = real( get_rad_step_size(), r8 )

      ! constant allometric parameters
      taper = 200._r8
      stocking = 1000._r8

      ! convert from stems/ha -> stems/m^2
      stocking = stocking / 10000._r8

      ! patch loop
      do fp = 1,num_soilp
         p = filter_soilp(fp)
         c = patch%column(p)
         g = patch%gridcell(p)

         if (ivt(p) /= noveg) then

            tlai_old = tlai(p) ! n-1 value
            tsai_old = tsai(p) ! n-1 value

            ! update the leaf area index based on leafC and SLA
            ! Eq 3 from Thornton and Zimmerman, 2007, J Clim, 20, 3902-3923. 
            if (dsladlai(ivt(p)) > 0._r8) then
               tlai(p) = (slatop(ivt(p))*(exp(leafc(p)*dsladlai(ivt(p))) - 1._r8))/dsladlai(ivt(p))
            else
               tlai(p) = slatop(ivt(p)) * leafc(p)
            end if
            tlai(p) = max(0._r8, tlai(p))

            ! update the stem area index and height based on LAI, stem mass, and veg type.
            ! With the exception of htop for woody vegetation, this follows the DGVM logic.

            ! tsai formula from Zeng et. al. 2002, Journal of Climate, p1835 (see notes)
            ! Assumes doalb time step .eq. CLM time step, SAI min and monthly decay factor
            ! alpha are set by PFT, and alpha is scaled to CLM time step by multiplying by
            ! dt and dividing by dtsmonth (seconds in average 30 day month)
            ! tsai_min scaled by 0.5 to match MODIS satellite derived values
            if (ivt(p) == nc3crop .or. ivt(p) == nc3irrig_dri .or. ivt(p) == nc3irrig_spr .or. ivt(p) == nc3irrig_fld) then ! generic crops

               tsai_alpha = 1.0_r8-1.0_r8*dt/dtsmonth
               tsai_min = 0.1_r8
            else
               tsai_alpha = 1.0_r8-0.5_r8*dt/dtsmonth
               tsai_min = 1.0_r8
            end if
            tsai_min = tsai_min * 0.5_r8
            tsai(p) = max(tsai_alpha*tsai_old+max(tlai_old-tlai(p),0._r8),tsai_min)

            if (woody(ivt(p)) == 1._r8) then

               ! trees and shrubs

               ! if shrubs have a squat taper 
               if (ivt(p) >= nbrdlf_evr_shrub .and. ivt(p) <= nbrdlf_dcd_brl_shrub) then
                  taper = 10._r8
                  ! otherwise have a tall taper
               else
                  taper = 200._r8
               end if

               ! trees and shrubs for now have a very simple allometry, with hard-wired
               ! stem taper (height:radius) and hard-wired stocking density (#individuals/area)
               if (use_cndv) then

                  if (fpcgrid(p) > 0._r8 .and. nind(p) > 0._r8) then

                     stocking = nind(p)/fpcgrid(p) !#ind/m2 nat veg area -> #ind/m2 patch area
                     htop(p) = allom2(ivt(p)) * ( (24._r8 * deadstemc(p) / &
                          (SHR_CONST_PI * stocking * dwood(ivt(p)) * taper))**(1._r8/3._r8) )**allom3(ivt(p)) ! lpj's htop w/ cn's stemdiam

                  else
                     htop(p) = 0._r8
                  end if

               else
                  !correct height calculation if doing accelerated spinup
                  if (spinup_state == 2) then
                    htop(p) = ((3._r8 * deadstemc(p) * 10._r8 * taper * taper)/ &
                         (SHR_CONST_PI * stocking * dwood(ivt(p))))**(1._r8/3._r8)
                  else
                    htop(p) = ((3._r8 * deadstemc(p) * taper * taper)/ &
                         (SHR_CONST_PI * stocking * dwood(ivt(p))))**(1._r8/3._r8)
                  end if

               endif

               ! Peter Thornton, 5/3/2004
               ! Adding test to keep htop from getting too close to forcing height for windspeed
               ! Also added for grass, below, although it is not likely to ever be an issue.
               htop(p) = min(htop(p),(forc_hgt_u_patch(p)/(displar(ivt(p))+z0mr(ivt(p))))-3._r8)

               ! Peter Thornton, 8/11/2004
               ! Adding constraint to keep htop from going to 0.0.
               ! This becomes an issue when fire mortality is pushing deadstemc
               ! to 0.0.
               htop(p) = max(htop(p), 0.01_r8)

               hbot(p) = max(0._r8, min(3._r8, htop(p)-1._r8))

            else if (ivt(p) >= npcropmin) then ! prognostic crops

               if (tlai(p) >= laimx(ivt(p))) peaklai(p) = 1 ! used in CNAllocation

               if (ivt(p) == ntmp_corn .or. ivt(p) == nirrig_dri_tmp_corn .or. ivt(p) == nirrig_spr_tmp_corn .or. ivt(p) == nirrig_fld_tmp_corn .or. &
                   ivt(p) == ntrp_corn .or. ivt(p) == nirrig_dri_trp_corn .or. ivt(p) == nirrig_spr_trp_corn .or. ivt(p) == nirrig_fld_trp_corn .or. &
                   ivt(p) == nsugarcane .or. ivt(p) == nirrig_dri_sugarcane .or. ivt(p) == nirrig_spr_sugarcane .or. ivt(p) == nirrig_fld_sugarcane .or. &
                   ivt(p) == nmiscanthus .or. ivt(p) == nirrig_dri_miscanthus .or. ivt(p) == nirrig_spr_miscanthus .or. ivt(p) == nirrig_fld_miscanthus .or. &
                   ivt(p) == nswitchgrass .or. ivt(p) == nirrig_dri_switchgrass .or. ivt(p) == nirrig_spr_switchgrass .or. ivt(p) == nirrig_fld_switchgrass) then
                  tsai(p) = 0.1_r8 * tlai(p)
               else
                  tsai(p) = 0.2_r8 * tlai(p)
               end if

               ! "stubble" after harvest
               if (harvdate(p) < 999 .and. tlai(p) == 0._r8) then
                  tsai(p) = 0.25_r8*(1._r8-farea_burned(c)*0.90_r8)    !changed by F. Li and S. Levis
                  htmx(p) = 0._r8
                  peaklai(p) = 0
               end if
               !if (harvdate(p) < 999 .and. tlai(p) > 0._r8) write(iulog,*) 'CNVegStructUpdate: tlai>0 after harvest!' ! remove after initial debugging?

               ! canopy top and bottom heights
               htop(p) = ztopmx(ivt(p)) * (min(tlai(p)/(laimx(ivt(p))-1._r8),1._r8))**2
               htmx(p) = max(htmx(p), htop(p))
               htop(p) = max(0.05_r8, max(htmx(p),htop(p)))
               hbot(p) = 0.02_r8

            else ! generic crops and ...

               ! grasses

               ! height for grasses depends only on LAI
               htop(p) = max(0.25_r8, tlai(p) * 0.25_r8)

               htop(p) = min(htop(p),(forc_hgt_u_patch(p)/(displar(ivt(p))+z0mr(ivt(p))))-3._r8)

               ! Peter Thornton, 8/11/2004
               ! Adding constraint to keep htop from going to 0.0.
               htop(p) = max(htop(p), 0.01_r8)

               hbot(p) = max(0.0_r8, min(0.05_r8, htop(p)-0.20_r8))
            end if

         else

            tlai(p) = 0._r8
            tsai(p) = 0._r8
            htop(p) = 0._r8
            hbot(p) = 0._r8

         end if

         ! adjust lai and sai for burying by snow. 
         ! snow burial fraction for short vegetation (e.g. grasses) as in
         ! Wang and Zeng, 2007.
         if (ivt(p) > noveg .and. ivt(p) <= nbrdlf_dcd_brl_shrub ) then
            ol = min( max(snow_depth(c)-hbot(p), 0._r8), htop(p)-hbot(p))
            fb = 1._r8 - ol / max(1.e-06_r8, htop(p)-hbot(p))
         else
            fb = 1._r8 - max(min(snow_depth(c),0.2_r8),0._r8)/0.2_r8   ! 0.2m is assumed
            !depth of snow required for complete burial of grasses
         endif

         elai(p) = max(tlai(p)*fb, 0.0_r8)
         esai(p) = max(tsai(p)*fb, 0.0_r8)

         ! Fraction of vegetation free of snow
         if ((elai(p) + esai(p)) > 0._r8) then
            frac_veg_nosno_alb(p) = 1
         else
            frac_veg_nosno_alb(p) = 0
         end if

      end do

    end associate 

 end subroutine CNVegStructUpdate

end module CNVegStructUpdateMod
